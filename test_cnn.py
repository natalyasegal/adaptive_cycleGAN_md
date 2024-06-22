#python test_cnn.py --batch_size 32 --test_data_path "out" --pretrained_model_path "efficientnet_b7_binary_classification.pth"

import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import models.cnn.efficientnet_b7_pt as en
from models.cnn.efficientnet_b7_pt import train_model, validate_model
from models.cnn.evaluation import calculate_metrics_test, plot_roc_curve


# Function to prepare data
def prepare_test_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Function to load the model
def load_model(model_path, num_classes=2):
    model = EfficientNet.from_pretrained('efficientnet-b7')
    model._conv_stem.in_channels = 1  # Change input channels to 1
    model._conv_stem.weight = nn.Parameter(model._conv_stem.weight.sum(dim=1, keepdim=True))  # Adjust the weights to work with 1 channel
    model._fc = nn.Linear(model._fc.in_features, num_classes)  # Adjust the final layer for binary classification
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to test the model
def test_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            #print(outputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
    return all_labels, all_predictions, all_probs

# Main function to run the test and evaluation
def main(args):
    test_loader = prepare_test_data(args.test_data_path, args.batch_size)
    model = load_model(args.pretrained_model_path, num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, predictions, probs = test_model(model, test_loader, device)

    calculate_metrics_test(labels, predictions, probs, th1 = 0.8)
    plot_roc_curve(labels, probs)
  
    print(f"Total samples: {len(labels)}")
    print(f"Samples per class: {dict(zip(*np.unique(labels, return_counts=True)))}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size',
                        help='batch size.',
                        type=int,
                        default=32)
    parser.add_argument('--test_data_path',
                        help='Path to the test dataset',
                        type=str,
                        default="out" #"datasets/ed/ed1_shifted_t9_09_20"#"datasets/ed/ed1_shifted_t9_085_1_20" #'datasets/ed/ed1_shifted_t5_20/' 
                       ) #ED4
    parser.add_argument('--pretrained_model_path',
                        help='Path to the trained model including name',
                        type=str,
                        default='efficientnet_b7_binary_classification.pth')

    args = parser.parse_args()
    main(args)
