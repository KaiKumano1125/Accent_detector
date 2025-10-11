#data_loader.py
import os
import librosa
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Daataset
from sklearn.preprocessing import LabelEncoder

class AudioDataset(Dataset):
    def __init__(self, csv_path= 'data/speakers_all.csv', audio_dir = 'data/recordings',n_mfcc =40, sr=16000):
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.sr = sr

        self.le = LabelEncoder()
        self.df['label'] = self.le.fit_transform(self.df['country'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['file'])

        y, sr = librosa.load(file_path, sr=self.sr)

        msfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        msfcc_tensor = torch.tensor(msfcc, dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.long)

        return msfcc_tensor, label
    
    def num_classes(self):
        return len(self.le.classes_)
    
    def save_label_encoder(self, path='label_encoder.npy'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.le.classes_)

def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_len = max(input.shape[-1] for input in inputs)

    padded_inputs = torch.zeros(len(inputs), inputs[0].shape[0], max_len)
    for i, input in enumerate(inputs):
        padded_inputs[i, :, :input.shape[-1]] = input

    return padded_inputs, torch.stack(labels)
