# Accent Classifier using PyTorch

A deep learning project that classifies **English accents by country** using the [Speech Accent Archive dataset](https://www.kaggle.com/datasets/rtatman/speech-accent-archive).  
Built with **PyTorch**, **Librosa**, and **TensorBoard** for training visualisation.

---

## Features
- Preprocesses and filters the Kaggle speech accent dataset
- Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from `.mp3` or `.wav` files
- Trains a lightweight **CNN-based accent classifier**
- Logs loss and accuracy in **TensorBoard**
- Supports GPU acceleration (CUDA)
- Automatically skips missing files and pads variable-length audio

---

## Tech Stack
- **Python 3.10+**
- **PyTorch**
- **Librosa**
- **Scikit-learn**
- **TensorBoard**

---
## Usae of each file
dataset_loader.py
-Loads the dataset and metadata from speaker_information.csv
-Extracts MFCC features from audio files using Librosa
-Handles missing or invalid files automatically
-Provides AccentDataset and collate_fn for PyTorch dataloaders

train.py
-Trains the CNN model on MFCC features
-Splits dataset into train / validation / test
-Logs loss and accuracy with TensorBoard
-Saves trained model weights (models/accent_cnn.pt)
-Saves label encoder classes (features/label_classes.npy)

inference.py
-Loads the trained model and label classes
-Takes an input .mp3 file
-Predicts the speaker’s country/accent
---

## Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/KaiKumano1125/accent_classifier.git
cd accent_classifier
pip install -r requirements.txt

