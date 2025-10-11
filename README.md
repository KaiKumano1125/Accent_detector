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

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/accent_classifier.git
   cd accent_classifier
