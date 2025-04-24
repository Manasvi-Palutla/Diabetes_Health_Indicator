# Diabetes Health Indicator - MLP Classifier

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to classify diabetes health indicators from the BRFSS2015 dataset. It handles class imbalance using SMOTE and evaluates performance with accuracy, loss graphs, and a confusion matrix.

---

## Dataset

- **Source**: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **Target Variable**: `Diabetes_012`  
  - `0 = No Diabetes`  
  - `1 = Pre-Diabetes`  
  - `2 = Diabetes`

---

##  Model Architecture & Training Details

### Multilayer Perceptron (MLP) Model
- **Input Layer**: Matches 21 input health indicators.
- **Hidden Layers**:  
  - 512 → 256 → 128 → 64 fully connected neurons  
  - Dropout rate of 0.3 applied after each hidden layer
- **Output Layer**: 3 neurons for multi-class classification
- **Loss Function**: Weighted CrossEntropyLoss to account for class imbalance
- **Optimizer**: Adam (adaptive learning rate)
- **Learning Rate Scheduler**: ReduceLROnPlateau with factor 0.5 on validation loss plateau

### Training Strategy
- **Epochs**: 150  
- **Batch Size**: 64  
- **Training Progress**:
  - **Epoch 1**: Train Loss: 0.7543, Val Loss: 0.6858  
  - **Epoch 20**: Train Acc: 82.6%, Val Acc: 84.25%  
  - **Final Epoch**: Train Acc: 91.18%, Val Acc: 90.28%

### Validation & Early Stopping
- Validation loss monitored each epoch
- Scheduler adjusted learning rate to ensure stable convergence
- Early stopping logic applied based on validation plateau

### Model Evaluation
- **Accuracy**: 90%
- **Metrics**:
  - High precision and recall for Class 0 and 1
  - Minor challenges in Class 2 classification (as expected from imbalanced data)
- **Confusion Matrix**: Visualized true vs predicted class distribution

### Visualizations
![Loss Curve](results/loss_curve.png)
*Training and Validation Loss over Epochs*

![Accuracy Curve](results/accuracy_curve.png)
*Training and Validation Accuracy over Epochs*

![Confusion Matrix](results/conf_matrix.png)
*Confusion Matrix showing prediction distribution*

---

##  Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt



