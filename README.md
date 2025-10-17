# Accent Classifier

A deep learning project that classifies **English accents by country** using the [Speech Accent Archive dataset](https://www.kaggle.com/datasets/rtatman/speech-accent-archive).  
Built with **PyTorch**, **Librosa**, and **TensorBoard** for training visualisation.

---

##  Features
- **Dataset Handling:** Loads `tw_food_101` with automatic class detection (101 folders)
- **Model Training:** Fine-tunes `ResNet-18` with Adam optimizer and Cross-Entropy loss
- **Preprocessing:** Resizes and normalizes images to `224×224`
- **Acceleration:** Full GPU (CUDA) support for efficient training
- **Monitoring:** Integrated TensorBoard logging for loss and accuracy
- **Model Export:** Saves trained weights to `src/output/tw_food101_best.pth`
- **Inference:** Includes a simple script (`inference.py`) for single-image prediction


---

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch
- **Audio/Signal Processing:** Librosa
- **Machine Learning Utilities:** Scikit-learn
- **Logging & Visualisation:** TensorBoard


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
```

##Author
Kai Kumano 

