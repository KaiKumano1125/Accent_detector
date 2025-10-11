import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from data_loader import AudioDataset, collate_fn
from tqdm import tqdm
import os
from datetime import datetime

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
    
def train_model(epochs=20, batch_size=32, lr = 1e-3):
    print("Loading dataset...")
    dataset = AudioDataset(csv_path='data/speakers_all.csv', audio_dir='data/recordings')

    dataset.save_label_encoder()

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = AccentCNN(num_classes = dataset.num_classes()).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for mfccs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            mfccs, labels = mfccs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for mfccs, labels in val_loader:
                mfccs, labels = mfccs.to(device), labels.to(device)
                outputs = model(mfccs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    #save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/accent_cnn.pt')
    print("Model saved to models/accent_cnn.pt")
    writer.close()

if __name__ == "__main__":
    train_model(epochs=20, batch_size=32, lr=1e-3)
