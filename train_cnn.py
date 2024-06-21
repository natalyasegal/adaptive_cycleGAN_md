import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from torchvision import datasets, transforms#, models
from torch.utils.data import DataLoader
import models.cnn.efficientnet_b7_pt as en

# Main function to run the training and validation
def main(args):
    data_dir = './datasets/ed/B_to/'  # Path to the dataset
    batch_size = 32
    num_epochs = args.epochs
    val_split = 0.2 #0.001 #on almost all of the data #0.2
    model_save_path = 'efficientnet_b7_binary_classification.pth'

    train_loader, val_loader = en.prepare_data(data_dir, batch_size, val_split=val_split)
    model = en.create_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    labels, predictions, probs = en.validate_model(model, val_loader, criterion, device)
    en.save_model(model, model_save_path)
    if val_split > 0.1:
      en.calculate_metrics(labels, predictions, probs)
      en.plot_roc_curve(labels, probs)

    print(f"Total samples: {len(labels)}")
    print(f"Samples per class: {dict(zip(*np.unique(labels, return_counts=True)))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=10)
    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=9)  
    parser.add_argument('--dataset_path',
                        help='from dataset path, dataset with the source distribution',
                        type=str,
                        default='./datasets/ed/A_from')
    

    args = parser.parse_args()
    main(args)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    labels, predictions, probs = validate_model(model, val_loader, criterion, device)
    save_model(model, model_save_path)
    if val_split > 0.1:
      calculate_metrics(labels, predictions, probs)
      plot_roc_curve(labels, probs)

    print(f"Total samples: {len(labels)}")
    print(f"Samples per class: {dict(zip(*np.unique(labels, return_counts=True)))}")

if __name__ == "__main__":
    main()
