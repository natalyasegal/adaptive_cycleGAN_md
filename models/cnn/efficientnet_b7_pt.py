import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt


# Function to prepare data
def prepare_data(data_dir, batch_size=32, val_split=0.2):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Function to define the model
def create_model(num_classes=2):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._conv_stem.in_channels = 1  # Change input channels to 1
    model._conv_stem.weight = nn.Parameter(model._conv_stem.weight.sum(dim=1, keepdim=True))  # Adjust the weights to work with 1 channel
    model._fc = nn.Linear(model._fc.in_features, num_classes)  # Adjust the final layer for binary classification
    return model

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")


# Function to validate the model
def validate_model(model, val_loader, criterion, device):
    model.to(device)
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    val_loss /= len(val_loader)
    val_acc = correct / total

    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    return all_labels, all_predictions, all_probs


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("Model saved successfully.")


