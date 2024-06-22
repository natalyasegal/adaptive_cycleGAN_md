import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from torchvision import datasets, transforms#, models
from torch.utils.data import DataLoader
from models.cycle_gan.adaptive_cycle_gan_pt import config_a_cgan, train_a_cgan
from models.cycle_gan.inference import batch_process_images

def main(args):
  config = config_a_cgan()
  config.num_epochs = args.epochs   
  config.texture_loss_weight = args.texture_loss_weight

  # Transformation to convert images to grayscale and normalize them
  transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
  ])
  
  # Load datasets
  dataset_A = datasets.ImageFolder(args.from_dataset_path, transform=transform)
  dataset_B = datasets.ImageFolder(args.to_dataset_path, transform=transform)
  dataloader_A = DataLoader(dataset_A, batch_size=1, shuffle=True)
  dataloader_B = DataLoader(dataset_B, batch_size=1, shuffle=True)

  train_a_cgan(config, dataloader_A, dataloader_B)

  batch_process_images(args.from_dataset_path, args.shifted_dataset_out_base_path)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=20)
    parser.add_argument('--texture_loss_weight',
                        help='texture loss weight in weighted sum of losses.',
                        type=int,
                        default=20)
    parser.add_argument('--random_seed',
                        help='seed for python random, used in shafling, does not affect division into train, validation and test',
                        type=int,
                        default=9)  
    parser.add_argument('--from_dataset_path',
                        help='from dataset path, dataset with the source distribution',
                        type=str,
                        default='./datasets/ed/A_from')
    parser.add_argument('--to_dataset_path',
                        help='to dataset path, dataset with the destination distribution',
                        type=str,
                        default='./datasets/ed/B_to/')
      parser.add_argument('--shifted_dataset_out_base_path',
                        help='to dataset path, dataset with the destination distribution',
                        type=str,
                        default='out')

    args = parser.parse_args()
    main(args)
