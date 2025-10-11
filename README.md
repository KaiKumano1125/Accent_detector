# Accent Classifier using PyTorch

A deep learning project that classifies **English accents by country** using the [Speech Accent Archive dataset](https://www.kaggle.com/datasets/rtatman/speech-accent-archive).  
Built with **PyTorch**, **Librosa**, and **TensorBoard** for training visualisation.

---

## Features
- Preprocesses and filters the Kaggle speech accent dataset
- Extracts MFCC (Mel-Frequency Cepstral Coefficients) features from `.mp3` files
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
##  Usage of Each File

###  `dataset_loader.py`
- Loads the dataset and metadata from **`speaker_information.csv`**
- Extracts **MFCC (Mel-Frequency Cepstral Coefficients)** features from audio files using *Librosa*
- Automatically skips missing or corrupted files
- Defines the `AccentDataset` class and `collate_fn` function for PyTorch dataloaders

---

###  `train.py`
- Trains the CNN model using extracted MFCC features
- Splits the dataset into **training**, **validation**, and **test** sets
- Logs loss and accuracy metrics in **TensorBoard**
- Saves the trained model weights → `models/accent_cnn.pt`
- Saves label encoder classes → `features/label_classes.npy`

---

###  `inference.py`
- Loads the trained model and label classes from saved files
- Accepts an input audio file (`.mp3` or `.wav`)
- Extracts MFCC features and predicts the **speaker’s country/accent**
- Displays the predicted accent in the console output
---

## Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/KaiKumano1125/accent_classifier.git
cd accent_classifier
pip install -r requirements.txt

