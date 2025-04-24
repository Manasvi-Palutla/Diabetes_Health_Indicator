import torch
from data_loader import load_data
from dataloader_utils import create_dataloader
from model import MLP
from train import train_model
from evaluate import evaluate_model
from plot_utils import plot_loss, plot_accuracy
from config import batch_size, learning_rate
import torch.optim as optim
import torch.nn as nn

# Load and prepare data
file_path = "diabetes_012_health_indicators_BRFSS2015.csv"
X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

# Create DataLoaders
train_loader = create_dataloader(X_train, y_train, batch_size)
val_loader = create_dataloader(X_val, y_val, batch_size)
test_loader = create_dataloader(X_test, y_test, batch_size)

# Initialize model
model = MLP(X_train.shape[1])

# Class weights
unique_classes, counts = torch.unique(y_train, return_counts=True)
class_weights = 1.0 / counts.float()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# Train model
training_loss, validation_loss, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler)

# Evaluate model
evaluate_model(model, test_loader, ["Class 0", "Class 1", "Class 2"])

# Plot
plot_loss(training_loss, validation_loss)
plot_accuracy(train_accuracies, val_accuracies)