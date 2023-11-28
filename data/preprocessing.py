import os
import nltk
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

# ------------------Constants---------------------------------------------------
IMAGES_PATH = "data/images/images_normalized/"
PROJECTIONS_PATH = "data/indiana_projections.csv"
REPORTS_PATH = "data/indiana_reports.csv"

BATCH_SIZE = 16

SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = '<pad>'
# ------------------Constants---------------------------------------------------
projections = pd.read_csv(PROJECTIONS_PATH)
projections = projections.set_index('uid')

reports = pd.read_csv(REPORTS_PATH)
reports = reports.set_index('uid')

data = projections.copy().drop(['projection'], axis=1)
for uid in reports.index:
    report = reports.loc[uid]['findings']
    data.loc[uid, 'report'] = report if type(report) == str else 'No findings.'

def load_reports():
    return [reports.loc[uid, 'findings'] if type(reports.loc[uid, 'findings']) == str else 'no findings.' for uid in projections.index]

class ChestXrayDataset(Dataset):
    def __init__(self, uids, img_folder, pad_token=PAD_TOKEN, max_length=512, from_pretrained= False, transform=None, tokenizer=None):
        self.uids = uids
        self.transform = transform
        self.pad_token = pad_token
        self.max_length = max_length
        self.data = data.loc[uids, :]
        self.img_folder = img_folder
        self.tokenizer = tokenizer
        self.for_pretrained = from_pretrained          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, report = self.data.iloc[idx]
        img1 = load_image(os.path.join(self.img_folder, filename))
        img1 = self.transform(img1)
        if not self.for_pretrained:
            report = report = report if type(report) == str else 'no findings.'
            report, length = self.tokenizer.padding(report, self.max_length)
            len_mask = [False if i < length else True for i in range(self.max_length)]
            return img1, torch.tensor(report, dtype=torch.long), torch.BoolTensor(len_mask)
        return img1, report.lower()

    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_pad_token(self):
        return self.word2idx['<pad>']

    def get_eos_token(self):
        return self.word2idx['</s>']

    def get_report(self, uid, to_idx=False):
        report = reports.loc[uid]['findings']
        report = report if type(report) == str else 'No findings'
        return report

def create_dataloaders(uids, img_folder, max_length= 30, batch_size=16, transform=None, tokenizer=None):
    # Split UIDs into train, validation, and test sets
    train_uids, test_uids = train_test_split(uids, test_size=0.2, random_state=33)
    train_uids, val_uids = train_test_split(train_uids, test_size=0.25, random_state=33)  # 0.25 x 0.8 = 0.2

    # Create datasets
    train_data = ChestXrayDataset(train_uids, img_folder, max_length=max_length, transform=transform, tokenizer=tokenizer)
    val_data = ChestXrayDataset(val_uids, img_folder, max_length= max_length, transform=transform, tokenizer=tokenizer)
    test_data = ChestXrayDataset(test_uids, img_folder, max_length= max_length, transform=transform, tokenizer=tokenizer)

    return train_data, val_data, test_data
    # # Create dataloaders
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # return train_data, train_loader, val_data, val_loader, test_data, test_loader

def __padding_report__(report, eos_token, padding_token, padding_size=100):
    if len(report) > padding_size:
        pad_report = report[:padding_size-1]
        pad_report = list(pad_report) + [eos_token]
    else:
        pad_report = list(report) + [padding_token for _ in range(padding_size - len(report))]
    return pad_report

def print_sequence(sequence, idx2word):
    print(' '.join([idx2word[idx] for idx in sequence]))

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img
