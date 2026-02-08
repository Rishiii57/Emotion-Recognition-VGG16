# Facial Emotion Recognition using Transfer Learning (VGG16)

This project implements **facial emotion recognition** using **transfer learning with VGG16** on the **FER-2013** dataset.  
Unlike a CNN trained from scratch, this approach leverages **pretrained ImageNet features** to improve generalization on a noisy, low-resolution facial emotion dataset.

This repository focuses specifically on **transfer learning and fine-tuning strategies**.

---

## Dataset

- **FER-2013**
- 48×48 grayscale facial images
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

---

## Motivation

FER-2013 is a challenging dataset due to:
- low image resolution
- label noise
- subtle inter-class differences
- class imbalance

Training a CNN from scratch often leads to overfitting and limited generalization.  
To address this, **transfer learning** is applied using a pretrained **VGG16** model.

---

## Methodology

### Model Architecture
- **Base model:** VGG16 pretrained on ImageNet
- Top (fully connected) layers removed
- Global Average Pooling applied
- Custom classification head added for 7 emotion classes

### Fine-Tuning Strategy
- Most VGG16 layers frozen
- Higher convolutional layers unfrozen for fine-tuning
- Very low learning rate used to preserve pretrained features
- Early stopping applied to prevent overfitting

### Preprocessing
- Images resized to **224×224**
- Converted to **RGB**
- VGG16-specific preprocessing applied (`preprocess_input`)

---

## Expected Performance

Transfer learning models such as **VGG16** are known to outperform CNNs trained from scratch on FER-2013.  
Based on prior research and common experimental results, validation accuracy is **typically observed in the 60–70% range**, representing a significant improvement over baseline CNN approaches.

NOTE: Exact accuracy may vary depending on fine-tuning depth, learning rate, and training environment.

---

## Tools & Technologies

- Python
- TensorFlow / Keras
- VGG16 (ImageNet pretrained)
- NumPy
- Matplotlib
- Jupyter Notebook

See `requirements.txt` for dependencies.

---

## Training Environment

Due to the computational cost of fine-tuning large pretrained models, this project is intended to be trained on **GPU-enabled environments** such as **Google Colab**.

---

## Future Improvements

- Fine-tuning additional VGG16 blocks
- Trying alternative pretrained models (ResNet, EfficientNet)
- Applying class weighting to handle imbalance
- Adding face detection before emotion classification
- Comparing results with CNN trained from scratch

---

## Author

Rishi57
