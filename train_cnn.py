import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms#, models
from torch.utils.data import DataLoader
import models.cnn.efficientnet_b7_pt as en
from models.cnn.efficientnet_b7_pt import train_model, validate_model
from models.cnn.evaluation import calculate_metrics, plot_roc_curve
import numpy as np

# Main function to run the training and validation
def main(args):
    val_split = 0.2
    train_loader, val_loader = en.prepare_data(args.dataset_path, args.batch_size, val_split=val_split)
    model = en.create_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, criterion, optimizer, device, args.epochs)
    labels, predictions, probs = validate_model(model, val_loader, criterion, device)
    en.save_model(model, args.model_save_path)
    if val_split > 0.1:
      calculate_metrics(labels, predictions, probs)
      plot_roc_curve(labels, probs)
    print(f"Total samples: {len(labels)}")
    print(f"Samples per class: {dict(zip(*np.unique(labels, return_counts=True)))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size.',
                        type=int,
                        default=32)
    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=9)  
    parser.add_argument('--dataset_path',
                        help='from dataset path, dataset with the source distribution',
                        type=str,
                        default='./datasets/ed/B_to/') #ED4
    parser.add_argument('--model_save_path',
                        help='path, including name, if the output model',
                        type=str,
                        default='efficientnet_b7_binary_classification.pth')

    args = parser.parse_args()
    main(args)

