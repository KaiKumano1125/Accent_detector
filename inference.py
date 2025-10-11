#inference
import torch
import torch.nn as nn
import librosa
import numpy as np
import os

class AccentCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) 
        )    
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
def predict_accent(audio_path, model_path='models/accent_cnn.pt', label_path='features/label_encoder.npy', n_mfcc=40, sr=16000):
    
    labels = np.load(label_path, allow_pickle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AccentCNN(num_classes=len(labels))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    y, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.ndim == 2:
        mfcc = np.expand_dims(mfcc, axis=0)  

    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs=model(x)
        pred_idx = torch.argmax(outputs, dim=1).item()

    pred_label = labels[pred_idx]
    print(f"Predicted accent: {pred_label}")
    return pred_label

if __name__ == "__main__":
    test_audio = 'data/recordings/japanese23.mp3'  
    if os.path.exists(test_audio):
        predict_accent(test_audio)
    else:
        print(f"Test audio file not found: {test_audio}")