import os
import nltk
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

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

# Global vocabulary
vocab = set()

class ChestXrayDataset(Dataset):
    def __init__(self, uids, img_folder, max_length=100, pad_token=PAD_TOKEN, transform=None):
        self.uids = uids
        self.transform = transform
        self.data = data.loc[uids, :]
        self.img_folder = img_folder
        self.vocab, self.word2idx, self.idx2word = self.__create_vocab(pad_token)

    def __create_vocab(self, pad_token):
        vocab = set()
        vocab.add(pad_token)
        for i in range(len(self.data)):
            _, report  = self.data.iloc[i, :]
            report = self.__sentence_tokenizer(report)
            vocab.update(report)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab, word2idx, idx2word

    def __sentence_tokenizer(self, text):
        return [SOS_TOKEN]+ nltk.word_tokenize(text) + [EOS_TOKEN]          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, report = self.data.iloc[idx]
        # report = self.__sentence_tokenizer(report)
        img1 = load_image(os.path.join(self.img_folder, filename))
        img1 = self.transform(img1)
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

def create_dataloaders(uids, img_folder, batch_size=16, transform=None):
    # Split UIDs into train, validation, and test sets
    train_uids, test_uids = train_test_split(uids, test_size=0.2, random_state=33)
    train_uids, val_uids = train_test_split(train_uids, test_size=0.25, random_state=33)  # 0.25 x 0.8 = 0.2

    # Create datasets
    train_data = ChestXrayDataset(train_uids, img_folder, transform=transform)
    val_data = ChestXrayDataset(val_uids, img_folder, transform=transform)
    test_data = ChestXrayDataset(test_uids, img_folder, transform=transform)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return train_data, train_loader, val_data, val_loader, test_data, test_loader

def padding_sequences(data, eos_token, padding_token, padding_size=100):
    sequences = []
    for _, _, report in data:
        if len(report) > padding_size:
            report = report[:padding_size-1]
            report = list(report) + [eos_token]
        else:
            report = list(report) + [padding_token for _ in range(padding_size - len(report)-1)]
            report = list(report) + [eos_token]
        # To tensor
        # report = torch.stack(report)
        sequences.append(list(report))
    return sequences

def bachify(data, batch_size):
    batches = []
    for i in range(0, len(data), batch_size):
        batches.append(data[i:i+batch_size])
    return batches

def print_sequence(sequence, idx2word):
    print(' '.join([idx2word[idx] for idx in sequence]))

def load_image(img_path):
    img = Image.open(img_path).convert('RGB')
    # img = transform(img)
    # img = img.unsqueeze(0)
    # imgr = resize(img, (224, 224))
    # img = np.expand_dims(img, axis=0)
    return img

def img2tensor(img):
    img_tensor = transforms.ToTensor()(img.astype(np.float32))
    img_tensor = img_tensor.permute(1, 2, 0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor