import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder
import math

__all__ = ['MMC']


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)


class ProjectionHead(nn.Module):
    """
    单模态唯一的投影头：同时服务于intra和inter对比学习
    """

    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 out_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):

        return self.projection(x)


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


class UnifiedEvidential_AdaPCL(nn.Module):
    """
    Unified Evidential Adaptive Prediction-Guided Contrastive Learning (Final)

    Key properties:
    - Direction: soft correctness (P(y))
    - Magnitude: evidential reliability (EDL)
    - Geometry: symmetric / directional alignment + conflict-aware FF slack
    """

    def __init__(
        self,
        num_classes,
        temperature=0.07,
        lambda_align=1.0,
        lambda_ff=0.1,
        gamma=0.5,
        ema_beta=0.9,
        margin=0.3
    ):
        super().__init__()
        self.tau = temperature
        self.num_classes = num_classes

        self.lambda_align = lambda_align
        self.lambda_ff = lambda_ff

        self.gamma = gamma
        self.ema_beta = ema_beta
        self.margin = margin

        self.register_buffer("ema_rel_a", torch.tensor(0.0))
        self.register_buffer("ema_rel_b", torch.tensor(0.0))

    def get_evidential_reliability(self, logits, ema_buffer):
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)

        uncertainty = self.num_classes / (S ** self.gamma + 1e-6)
        reliability = (1.0 - uncertainty).clamp(0.0, 1.0)

        if self.training:
            batch_mean = reliability.mean().detach()
            ema_buffer.mul_(self.ema_beta).add_(batch_mean * (1 - self.ema_beta))

        return (0.5 * reliability + 0.5 * ema_buffer).squeeze(1).detach()

    def scale_loss(self, loss_vec, weight_vec):

        s = weight_vec.sum()
        if s < 1e-6:
            return 0.0
        return (loss_vec * weight_vec).sum() / s

    def forward(self, feat_a, feat_b, logit_a, logit_b, labels):
        B = feat_a.size(0)
        feat_a = F.normalize(feat_a, dim=1)
        feat_b = F.normalize(feat_b, dim=1)

        prob_a = F.softmax(logit_a, dim=1)[torch.arange(B), labels]
        prob_b = F.softmax(logit_b, dim=1)[torch.arange(B), labels]

        m_tt = prob_a * prob_b
        m_tf = prob_a * (1 - prob_b)
        m_ft = (1 - prob_a) * prob_b
        m_ff = (1 - prob_a) * (1 - prob_b)


        r_a = self.get_evidential_reliability(logit_a, self.ema_rel_a)
        r_b = self.get_evidential_reliability(logit_b, self.ema_rel_b)


        sim_ab = torch.matmul(feat_a, feat_b.T) / self.tau
        sim_ba = sim_ab.T

        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
        neg_mask = (~label_mask) | torch.eye(B, device=feat_a.device).bool()

        sim_ab = sim_ab.masked_fill(~neg_mask, -1e9)
        sim_ba = sim_ba.masked_fill(~neg_mask, -1e9)

        targets = torch.arange(B, device=feat_a.device)


        L_sym = 0.5 * (
            F.cross_entropy(sim_ab, targets, reduction="none") +
            F.cross_entropy(sim_ba, targets, reduction="none")
        )


        w_tt = torch.pow(m_tt * r_a * r_b + 1e-8, 1.0 / 3.0)  # 3个主要维度：准确、可靠A、可靠B

        loss_tt_final = self.scale_loss(L_sym, w_tt)


        L_tf = F.cross_entropy(
            torch.matmul(feat_b, feat_a.detach().T) / self.tau,
            targets, reduction="none"
        )

        L_ft = F.cross_entropy(
            torch.matmul(feat_a, feat_b.detach().T) / self.tau,
            targets, reduction="none"
        )



        w_tf = torch.pow(m_tf * r_a + 1e-8, 1.0 / 2.0)
        w_ft = torch.pow(m_ft * r_b + 1e-8, 1.0 / 2.0)

        loss_tf_final = self.scale_loss(L_tf, w_tf)
        loss_ft_final = self.scale_loss(L_ft, w_ft)


        cos_sim = F.cosine_similarity(feat_a, feat_b, dim=1)

        p_a = F.softmax(logit_a, dim=1)
        p_b = F.softmax(logit_b, dim=1)
        conflict = torch.norm(p_a - p_b, p=2, dim=1)

        r_max = torch.max(r_a, r_b)

        loss_ff =  m_ff *(1 - r_max)
        loss_ff_final = (loss_ff).mean()

        total_loss = self.lambda_align * (loss_tt_final + loss_tf_final + loss_ft_final) + \
                     self.lambda_ff * loss_ff_final

        return total_loss


class MMC(nn.Module):
    def __init__(self, args):
        super(MMC, self).__init__()
        # text subnets
        self.args = args
        if self.args.mmc not in ['T']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.image_classfier = Classifier(args.img_dropout, args.img_out, args.post_dim, args.output_dim)
        if self.args.mmc not in ['V']:
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
            self.text_classfier = Classifier(args.text_dropout, args.text_out, args.post_dim, args.output_dim)
        self.mm_classfier = Classifier(args.mm_dropout, args.text_out + args.img_out, args.post_dim, args.output_dim)

        self.mm_contra_classfier = Classifier(args.mm_dropout, args.text_out, args.post_dim, args.output_dim)

        self.unified_evi = UnifiedEvidential_AdaPCL(args.output_dim)

        self.text_contrast_head = ProjectionHead(in_dim=args.text_out)
        self.image_contrast_head = ProjectionHead(in_dim=args.img_out)


    def forward(self, text=None, image=None, data_list=None, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        text = self.text_encoder(text=text)
        image = torch.squeeze(image, 1)
        image = self.image_encoder(pixel_values=image)

        contrast_text = self.text_contrast_head(text[:, 0, :])
        contrast_image = self.image_contrast_head(image[:, 0, :])

        output_text = self.text_classfier(text[:, 0, :])
        output_image = self.image_classfier(image[:, 0, :])


        # 1.concat
        fusion = torch.cat([text[:, 0, :], image[:, 0, :]], dim=-1)
        output_mm = self.mm_classfier(fusion)

        if infer:
            return output_mm

        MMLoss_m = torch.mean(criterion(output_mm, label))


        MMLoss_text = torch.mean(criterion(output_text, label))
        MMLoss_image = torch.mean(criterion(output_image, label))



        mmcLoss = self.unified_evi(contrast_text, contrast_image, output_text, output_image, label)

        # 只计算单模态、融合损失
        MMLoss_sum = MMLoss_text + MMLoss_image + MMLoss_m + 5 * mmcLoss

        return MMLoss_sum, MMLoss_m, output_mm   # 总体损失，融合损失，融合输出

    def infer(self, text=None, image=None, data_list=None):
        MMlogit = self.forward(text, image, data_list, infer=True)
        return MMlogit


    def mmc_2(self, f0, f1, p0, p1, l):
        f0 = f0 / f0.norm(dim=-1, keepdim=True)
        f1 = f1 / f1.norm(dim=-1, keepdim=True)

        if p0 is not None:
            p0 = torch.argmax(F.softmax(p0, dim=1), dim=1)
            p1 = torch.argmax(F.softmax(p1, dim=1), dim=1)

        if l is None:
            return self.UnSupMMConLoss(f0, f1)
        elif p0 is None:
            return self.SupMMConLoss(f0, f1, l)
        else:
            return self.UniSMMConLoss(f0, f1, p0, p1, l)

    def info_nce_loss(self, emb1, emb2, temperature=0.07):
        """
        通用的InfoNCE损失（适配intra和inter对比）
        - emb1/emb2: 需对比的两组特征 [batch_size, out_dim]（已归一化）
        """
        emb1 = F.normalize(emb1, dim=-1)
        emb2 = F.normalize(emb2, dim=-1)

        batch_size = emb1.shape[0]
        # 计算相似度矩阵
        sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
        # 正样本：对角线（i和i匹配）
        labels = torch.arange(batch_size).to(emb1.device)
        # 交叉熵损失（双向对比可选，此处单方向即可）
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def UniSMMConLoss(self, feature_a, feature_b, predict_a, predict_b, labels, temperature=0.07):
        feature_a_ = feature_a.detach()
        feature_b_ = feature_b.detach()

        a_pre = predict_a.eq(labels)  # a True or not
        a_pre_ = ~a_pre
        b_pre = predict_b.eq(labels)  # b True or not
        b_pre_ = ~b_pre

        a_b_pre = torch.gt(a_pre | b_pre, 0)  # For mask ((P: TT, nP: TF & FT)=T, (N: FF)=F)
        a_b_pre_ = torch.gt(a_pre & b_pre, 0) # For computing nP, ((P: TT)=T, (nP: TF & FT, N: FF)=F)

        a_ = a_pre_ | a_b_pre_  # For locating nP not gradient of a
        b_ = b_pre_ | a_b_pre_  # For locating nP not gradient of b

        if True not in a_b_pre:
            a_b_pre = ~a_b_pre
            a_ = ~a_
            b_ = ~b_
        mask = a_b_pre.float()
#
        feature_a_f = [feature_a[i].clone() for i in range(feature_a.shape[0])]
        for i in range(feature_a.shape[0]):
            if not a_[i]:
                feature_a_f[i] = feature_a_[i].clone()
        feature_a_f = torch.stack(feature_a_f)

        feature_b_f = [feature_b[i].clone() for i in range(feature_b.shape[0])] # feature_b  # [[0,1]])
        for i in range(feature_b.shape[0]):
            if not b_[i]:
                feature_b_f[i] = feature_b_[i].clone()
        feature_b_f = torch.stack(feature_b_f)

        # compute logits
        logits = torch.div(torch.matmul(feature_a_f, feature_b_f.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)

        # compute log_prob
        exp_logits = torch.exp(logits-logits_max.detach())[0]
        mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())# + 1e-6

        return mean_log_pos

    def SupMMConLoss(self, feature_a, feature_b, labels, temperature=0.07):
        # compute the mask matrix
        labels = labels.contiguous().view(-1, 1)
        # mask = torch.eq(labels, labels.T).float() - torch.eye(feature_a.shape[0], feature_a.shape[0])
        mask = torch.eq(labels, labels.T).float()

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)

        return mean_log_pos.mean()

    def UnSupMMConLoss(self, feature_a, feature_b, temperature=0.07):

        # compute the mask matrix
        mask = torch.eye(feature_a.shape[0], dtype=torch.float32).to(self.args.device)

        # compute logits
        logits = torch.div(torch.matmul(feature_a, feature_b.T), temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_pos = -(mask * log_prob).sum(1) / mask.sum(1)
        mean_log_pos = mean_log_pos.mean()

        return mean_log_pos


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, post_dim, out_dim):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.post_layer_1 = LinearLayer(in_dim, post_dim)
        self.post_layer_2 = LinearLayer(post_dim, post_dim)
        self.post_layer_3 = LinearLayer(post_dim, out_dim)

    def forward(self, input):
        input_p1 = F.relu(self.post_layer_1(input), inplace=False)
        input_d = self.post_dropout(input_p1)
        input_p2 = F.relu(self.post_layer_2(input_d), inplace=False)
        output = self.post_layer_3(input_p2)
        return output






