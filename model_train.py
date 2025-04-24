import torch
from config import num_epochs
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    training_loss = []
    validation_loss = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            running_loss += loss.item()

        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        avg_loss = running_loss / len(train_loader)
        training_loss.append(avg_loss)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss = criterion(val_outputs, val_y)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_y.size(0)
                val_correct += (val_predicted == val_y).sum().item()
                val_running_loss += val_loss.item()

        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        avg_val_loss = val_running_loss / len(val_loader)
        validation_loss.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        scheduler.step(avg_val_loss)

    return training_loss, validation_loss, train_accuracies, val_accuracies

import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.hidden4 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.dropout(torch.relu(self.hidden2(x)))
        x = self.dropout(torch.relu(self.hidden3(x)))
        x = self.dropout(torch.relu(self.hidden4(x)))
        return self.output(x)