#data_loader.py
import os
import librosa
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class AudioDataset(Dataset):
    def __init__(self, csv_path= 'data/speakers_all.csv', audio_dir = 'data/recordings',n_mfcc =40, sr=16000):
        
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.n_mfcc = n_mfcc
        self.sr = sr

        if "file" in self.df.columns:
            file_col = "file"
        elif "filename" in self.df.columns:
            file_col = "filename"
        else:
            raise KeyError("CSV must contain 'file' or 'filename' column.")

        # --- Fix extensions and keep only existing audio ---
        valid_rows = []
        for _, row in self.df.iterrows():
            fname = str(row[file_col])
            # ensure extension
            if not fname.lower().endswith((".wav", ".mp3")):
                fname += ".mp3"
            path = os.path.join(audio_dir, os.path.basename(fname))
            if os.path.exists(path):
                row[file_col] = os.path.basename(fname)
                valid_rows.append(row)
            else:
                print(f"⚠️  Skipping missing file: {fname}")

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # --- Encode labels (country) ---
        self.le = LabelEncoder()
        self.df["label"] = self.le.fit_transform(self.df["country"])
        print(f"✅ Loaded {len(self.df)} valid audio samples after filtering.")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']  
        if not filename.lower().endswith('.mp3'):
             filename = filename + '.mp3'

        filename = os.path.basename(filename)
        file_path = os.path.join(self.audio_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
             
        y, sr = librosa.load(file_path, sr=self.sr)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        if mfcc.ndim == 2:
             mfcc = np.expand_dims(mfcc, axis=0)
             mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
             label_tensor = torch.tensor(row["label"], dtype=torch.long)
        return mfcc_tensor, label_tensor
    
    def num_classes(self):
        return len(self.le.classes_)

    def save_label_encoder(self, path='features/label_encoder.npy'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.le.classes_)

def collate_fn(batch):
    """Pads MFCC tensors along the time dimension for batching."""
    inputs, labels = zip(*batch)
    max_len = max(x.shape[-1] for x in inputs)
    n_mfcc = inputs[0].shape[1]
    batch_size = len(inputs)

    padded = torch.zeros(batch_size, 1, n_mfcc, max_len)

    for i, x in enumerate(inputs):
        if x.ndim == 2:  
            x = x.unsqueeze(0)
        time_len = x.shape[-1]
        padded[i, :, :, :time_len] = x

    return padded, torch.stack(labels)

