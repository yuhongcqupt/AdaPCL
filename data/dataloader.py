import re
import os
import time
import argparse
import random
import logging
import json
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

from model.TextEncoder import TextEncoder
from model.ImageEncoder import ImageEncoder
import pickle

import re
import html
import string
import emoji
import unicodedata
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


__all__ = ['MMDataLoader']

logger = logging.getLogger('MMC')

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

# Data Aug
def get_transforms():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
        ]
    )


def remove_tags(text):
    return TAG_RE.sub('', text)


# vec_load_image = np.vectorize(load_image, signature='()->(r,c,d),(s)')
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = sentence.lower()
    return sentence


def format_txt_file(content):
    for c in '<>/\\+=-_[]{}\'\";:.,()*&^%$#@!~`':
        content = content.replace(c, ' ')
    content = re.sub("\s\s+" , ' ', content)
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = re.sub(r"\s+[a-zA-Z]\s+", ' ', content)
    return content.lower().replace("\n", " ")


label_n24news = { 'Health': 0,
                  'Books': 1,
                  'Science': 2,
                  'Art & Design': 3,
                  'Television': 4,
                  'Style': 5,
                  'Travel': 6,
                  'Media': 7,
                  'Movies': 8,
                  'Food': 9,
                  'Dance': 10,
                  'Well': 11,
                  'Real Estate': 12,
                  'Fashion & Style': 13,
                  'Economy': 14,
                  'Technology': 15,
                  'Sports': 16,
                  'Your Money': 17,
                  'Theater': 18,
                  'Education': 19,
                  'Opinion': 20,
                  'Automobiles': 21,
                  'Music': 22,
                  'Global Business': 23,
                  }

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()



def integrated_clean_tweet(text: str, is_bert: bool = True) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return "[EMPTY]"

    # 基础预处理
    text = html.unescape(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\n', ' ').replace('\t', ' ')

    # 处理 RT 标记
    if text.startswith('RT '):
        text = re.sub(r'^RT\s+@[A-Za-z0-9_]+:\s+', '', text)

    # 替换用户和链接
    text = re.sub(r'@[A-Za-z0-9_]+', '[USER]', text)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '[LINK]', text)

    # Hashtag 处理
    text = re.sub(r'#([A-Za-z0-9_]+)', r'\1', text)

    # Emoji 转换
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 重复字符缩减
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 标点符号处理
    if not is_bert:
        punc_pattern = re.compile('[%s]' % re.escape(string.punctuation))
        text = re.sub(punc_pattern, '', text)
    else:
        text = re.sub(r'[^0-9a-zA-Z\s\[\]:_?!\.,\']+', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip().lower()

    # 特殊情况检查
    if not text or text in ['[link]', '[user]', 'rt']:
        return "[EMPTY]"

    return text


class MMDataset(Dataset):
    def __init__(self, args, labels):
        self.args = args
        self.labels = labels
        self.save = []
        print(os.path.join(args.data_dir, labels))
        if args.dataset in ['Food101']:
            self.df = pd.read_csv(os.path.join(args.data_dir, labels),
                                  dtype={'id': str, 'text': str, 'annotation': str, 'label': int})

        elif args.dataset in ['N24News']:
            self.df = json.load(open(os.path.join(args.data_dir, labels), 'r', encoding='utf8'))

        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()

        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.max_length = args.max_length  # Setup according to the text
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.args.dataset in ['Food101']:
            id, text, annotation, label = self.df.loc[index]
            img_path = self.args.data_dir + '/images/' + self.labels[:-4] + '/' + annotation + '/' + id
            text_path = self.args.data_dir + '/texts_txt/' + annotation + '/' + id.replace(".jpg", ".txt")
            text = format_txt_file(open(text_path).read())

        elif self.args.dataset in ['N24News']:
            if self.args.text_type in ['headline']:
                text = self.df[index]['headline']
            elif self.args.text_type in ['caption']:
                text = self.df[index]['caption']
            elif self.args.text_type in ['abstract']:
                text = self.df[index]['abstract']
            else:
                text = self.df[index]['article']
                if self.args.text_encoder not in ['roberta_base']:
                    text = format_txt_file(text)
            img_path = self.args.data_dir + '/imgs/' + self.df[index]['image_id'] + '.jpg'
            label = label_n24news[self.df[index]['section']]

        # text -> text_token
        text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True,
                                     padding='max_length', return_tensors="pt")
        image = Image.open(os.path.join(img_path)).convert("RGB")
        image = self.transforms(image)
        img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values

        if 'roberta' in self.args.text_encoder:
            return img_inputs, text_tokens['input_ids'], 0, text_tokens['attention_mask'], label
        else:
            return img_inputs, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens[
                'attention_mask'], label


def MMDataLoader(args):
    # if args.dataset in ['Food101']:
    #     train_data_set = MMDataset(args, 'train.csv')
    #     train_set, valid_set = torch.utils.data.random_split(train_data_set, [len(train_data_set)-5000, 5000])
    #     test_set = MMDataset(args, 'test.csv')
    if args.dataset in ['Food101']:
        train_data_set = MMDataset(args, 'train.csv')
        test_set = MMDataset(args, 'test.csv')
        all_labels = train_data_set.df['label'].tolist()
        all_indices = list(range(len(train_data_set)))
        val_ratio = 0.1
        train_idx, valid_idx = train_test_split(
            all_indices,
            test_size=val_ratio,
            stratify=all_labels,
            random_state=42
        )
        train_set = Subset(train_data_set, train_idx)
        valid_set = Subset(train_data_set, valid_idx)

    elif args.dataset in ['N24News']:
        train_set = MMDataset(args, 'news/nytimes_train.json')
        valid_set = MMDataset(args, 'news/nytimes_dev.json')
        test_set = MMDataset(args, 'news/nytimes_test.json')

    elif args.dataset in ['tumemo']:
        meta_pkl_path = '/home/Data/TumEmo/TumEmo Original/tum_emo.pkl'
        data_dir = '/home/Data/TumEmo/TumEmo Original/all_data/'
        max_length = 128
        train_set = IT_Dataset(args, meta_pkl_path, data_dir, split='train', max_length=max_length)
        valid_set = IT_Dataset(args, meta_pkl_path, data_dir, split='val', max_length=max_length)
        test_set = IT_Dataset(args, meta_pkl_path, data_dir, split='test', max_length=max_length)

    elif args.dataset in ['fakeddit']:
        meta_pkl_path = '/home//Data/Fakeddit_MM/fakeddit_splits.pkl'
        data_dir = '/home/Data/Fakeddit_MM/merged_images/'
        max_length = 32

        train_set = IT_Dataset(args, meta_pkl_path, data_dir, split='train', max_length=max_length)
        valid_set = IT_Dataset(args, meta_pkl_path, data_dir, split='val', max_length=max_length)
        test_set = IT_Dataset(args, meta_pkl_path, data_dir, split='test', max_length=max_length)



    logger.info(f'Train Dataset: {len(train_set)}')
    logger.info(f'Valid Dataset: {len(valid_set)}')
    logger.info(f'Test Dataset: {len(test_set)}')

    if args.local_rank in [-1]:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=False, pin_memory=False, drop_last=True)
    else:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       sampler=train_sampler, pin_memory=False, drop_last=True)

    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                       shuffle=False, pin_memory=False, drop_last=True)

    return train_loader, valid_loader, test_loader


class IT_Dataset(Dataset):
    def __init__(
        self,
        args,
        meta_pkl,
        data_dir,
        split,
        max_length=128
    ):
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)

        self.data = meta["data"]
        self.indices = meta["splits"][split]

        self.data_dir = data_dir
        self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir,
                                          text_encoder=args.text_encoder).get_tokenizer()
        self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir,
                                            image_encoder=args.image_encoder).get_tokenizer()

        self.max_length = max_length
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.data[real_idx]

        # ===== text =====
        text_tokens = self.text_tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # ===== image =====

        img_path = os.path.join(self.data_dir, sample["filename"] + ".jpg")

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            print(img_path)
        image = self.transforms(image)
        image_inputs = self.image_tokenizer(image, return_tensors="pt").pixel_values

        file_name = sample["filename"]
        input_ids = text_tokens["input_ids"]
        attention_mask = text_tokens["attention_mask"]
        token_type_ids = text_tokens['token_type_ids']
        # pixel_values = image_inputs["pixel_values"] # [3, 224, 224]
        label = torch.tensor(sample["label"], dtype=torch.long)

        return image_inputs, input_ids, token_type_ids, attention_mask, label

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='MMC',
                        help='project name')
    parser.add_argument('--dataset', type=str, default='N24News',
                        help='support N24News/Food101/')
    parser.add_argument('--text_type', type=str, default='caption',
                        help='support headline/caption/abstract')
    parser.add_argument('--mmc', type=str, default='UniSMMC',
                        help='support UniSMMC/UnSupMMC/SupMMC')
    parser.add_argument('--mmc_tao', type=float, default=0.07,
                        help='use supervised contrastive loss or not')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--lr_mm', type=float, default=1e-3,
                        help='--lr_mm')
    parser.add_argument('--min_epoch', type=int, default=1,
                        help='min_epoch')
    parser.add_argument('--valid_step', type=int, default=50,
                        help='valid_step')
    parser.add_argument('--max_length', type=int, default=100,
                        help='max_length')
    parser.add_argument('--text_encoder', type=str, default='bert-base',
                        help='bert_base/roberta_base/bert_large')
    parser.add_argument('--image_encoder', type=str, default='vit-base',
                        help='vit_base/vit_large')
    parser.add_argument('--text_out', type=int, default=768,
                        help='text_out')
    parser.add_argument('--img_out', type=int, default=768,
                        help='img_out')
    parser.add_argument('--lr_mm_cls', type=float, default=1e-3,
                        help='--lr_mm_cls')
    parser.add_argument('--mm_dropout', type=float, default=0.0,
                        help='--mm_dropout')
    parser.add_argument('--lr_text_tfm', type=float, default=2e-5,
                        help='--lr_text_tfm')
    parser.add_argument('--lr_img_tfm', type=float, default=5e-5,
                        help='--lr_img_tfm')
    parser.add_argument('--lr_img_cls', type=float, default=1e-4,
                        help='--lr_img_cls')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5,
                        help='--lr_text_cls')
    parser.add_argument('--text_dropout', type=float, default=0.0,
                        help='--text_dropout')
    parser.add_argument('--img_dropout', type=float, default=0.1,
                        help='--img_dropout')
    parser.add_argument('--nplot', type=str, default='',
                        help='MTAV')
    parser.add_argument('--data_dir', type=str, default='/home/Data/',
                        help='support wmsa')
    parser.add_argument('--test_only', type=bool, default=False,
                        help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='/home/programFile/pre_train_model/',
                        help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='/home/programFile/AdaPCL/results/models',
                        help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='/home/programFile/AdaPCL/results/results',
                        help='path to save training results.')
    parser.add_argument('--fig_save_dir', type=str, default='/home/programFile/AdaPCL/results/imgs',
                        help='path to save figures.')
    parser.add_argument('--logs_dir', type=str, default='/home/programFile/AdaPCL/results/logs',
                        help='path to log results.')  # NO
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help='set seeds for multiple runs!')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = MMDataLoader(args)
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        image, input_ids, token_type, attention_mask,  label = batch

        print(input_ids.shape)
        print(image.shape)
        print(label)
