# Dogs vs Cats Classification — Vanilla CNN vs VGG16 Transfer Learning
### PL3 — Convolutional Neural Networks (Final Lab Notebook)

This notebook implements an end-to-end deep learning workflow on the **Dogs vs Cats** dataset using two models:

1. **A custom Vanilla CNN** (trained from scratch)  
2. **A fine-tuned VGG16** model using transfer learning  

The goal is to demonstrate the performance difference between building a small CNN versus using a powerful ImageNet-pretrained model.

## Project Objectives

This notebook fulfills the lab requirements by implementing:

- Dataset preparation using the **5,000-image subset**
- Exploratory Data Analysis (EDA)
- Model 1: **Vanilla CNN**
- Model 2: **VGG16 Transfer Learning**
- Use of **callbacks** (ModelCheckpoint + EarlyStopping)
- Evaluation of both models using:
  - Accuracy & loss  
  - Confusion matrix  
  - Classification report  
  - Precision–Recall curves  
  - Misclassified examples
- Side-by-side model comparison
- Final conclusions

## Folder Structure

```
.
├── dogs_vs_cats_5000/
│   ├── train/
│   ├── val/
│   └── test/
├── PL3_dogs_vs_cats_VanillaCNN_VGG16.ipynb
└── README.md
```

## Models Implemented

### **Vanilla CNN (Baseline Model)**  
A lightweight CNN designed and trained from scratch.  
Includes convolution, pooling, dropout, and dense layers.

Used to establish baseline performance.

### **VGG16 (Transfer Learning Model)**  
A powerful model built using:

- Pre-trained ImageNet convolutional base  
- Frozen feature extractor  
- Custom Dense classification head  
- Fine-tuning with callbacks  

This demonstrates how transfer learning improves performance with limited data.

## Evaluation Metrics

Both models were evaluated on:

- Test Accuracy  
- Test Loss  
- Confusion Matrix  
- Precision, Recall, F1-score  
- Precision–Recall (PR) Curves  
- Misclassified Sample Visualization  
- Combined PR Curve (CNN vs VGG16)

## Key Results

| **Metric** | **Vanilla CNN** | **VGG16 (Fine-Tuned)** |
|------------|------------------|--------------------------|
| **Test Accuracy** | 0.745 | 0.973 |
| **Test Loss** | 0.562 | 0.188 |
| **Precision (Cat/Dog)** | 0.78 / 0.72 | 0.97 / 0.98 |
| **Recall (Cat/Dog)** | 0.69 / 0.80 | 0.97 / 0.98 |
| **Confusion Matrix Errors** | Cat→Dog: 157, Dog→Cat: 98 | Cat→Dog: 17, Dog→Cat: 11 |
| **Misclassifications** | ~255 images | < 40 images |

**Conclusion:**  
VGG16 significantly outperforms the baseline CNN across all metrics.

## Visualizations Included

- Dataset sample grids  
- Class distribution graphs  
- Training & validation accuracy/loss curves  
- PR curve for CNN  
- PR curve for VGG16  
- Combined PR curve  
- Misclassified sample plots  
- Comparison table & final analysis  

## Final Conclusions

- A CNN trained from scratch performs reasonably well but has limited feature extraction ability.  
- Transfer learning with VGG16 delivers **dramatically better accuracy, precision, and recall**.  
- PR curves show VGG16 maintains high precision across almost all recall levels.  
- VGG16 is the stronger and more reliable model for real-world image classification tasks.

## Requirements

- Python 3.10+  
- TensorFlow / Keras  
- scikit-learn  
- matplotlib  
- Dogs vs Cats 5k dataset
