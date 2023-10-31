import os
import cv2
import nltk
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

nltk.download('punkt')

# ------------------Constants---------------------------------------------------
IMAGES_PATH = "data/images/images_normalized/"
PROJECTIONS_PATH = "data/indiana_projections.csv"
REPORTS_PATH = "data/indiana_reports.csv"

BATCH_SIZE = 16
# ------------------Constants---------------------------------------------------
projections = pd.read_csv(PROJECTIONS_PATH)
projections = projections.set_index('uid')

reports = pd.read_csv(REPORTS_PATH)
reports = reports.set_index('uid')

# Global vocabulary
vocab = set()

class ChestXrayDataset(Dataset):
    def __init__(self, projections, reports, uids, img_folder, pad_token='<pad>'):
        self.projections = projections
        self.reports = reports
        self.img_folder = img_folder
        self.data = self.__generate_data(uids)
        self.vocab, self.word2idx, self.idx2word = self.__create_vocab(pad_token)

    def __create_vocab(self, pad_token):
        vocab = set()
        vocab.add(pad_token)
        for _, _, report in self.data:
            vocab.update(report)
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for idx, word in enumerate(vocab)}
        return vocab, word2idx, idx2word

    def __sentence_tokenizer(self, text):
        return ['<s>']+ nltk.word_tokenize(text) + ['</s>']

    def __generate_data(self, uids):
        data = []
        for iter, uid in enumerate(uids):
            print(f'Generating data: {iter+1}/{len(uids)}', end='\r')
            try:
                # Sometimes the report is not available
                report = reports.loc[uid]['findings']
                report = self.__sentence_tokenizer(report)
                filename1, filename2 = self.projections.loc[uid]['filename']
                img1 = load_image(os.path.join(self.img_folder, filename1))
                img2 = load_image(os.path.join(self.img_folder, filename2))
            except:
                continue
            data.append((img1, img2, report))
        return data            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, report = self.data[idx]
        report = [self.word2idx[word] for word in report]
        # report = torch.LongTensor(report)
        return img1, img2, report

    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_pad_token(self):
        return self.word2idx['<pad>']

def create_dataloaders(uids, projections, reports, img_folder, batch_size=16):
    # Split UIDs into train, validation, and test sets
    train_uids, test_uids = train_test_split(uids, test_size=0.2, random_state=33)
    train_uids, val_uids = train_test_split(train_uids, test_size=0.25, random_state=33)  # 0.25 x 0.8 = 0.2

    # Create datasets
    train_data = ChestXrayDataset(projections, reports, train_uids, img_folder)
    val_data = ChestXrayDataset(projections, reports, val_uids, img_folder)
    test_data = ChestXrayDataset(projections, reports, test_uids, img_folder)

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return train_data, train_loader, val_data, val_loader, test_data, test_loader

def get_sequences(data, eos_token, padding_token, padding_size=100):
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
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    imgr = resize(img, (2048, 2048))
    imgr = np.expand_dims(imgr, axis=0)
    return imgr

def img2tensor(img):
    img_tensor = transforms.ToTensor()(img.astype(np.float32))
    img_tensor = img_tensor.permute(1, 2, 0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor