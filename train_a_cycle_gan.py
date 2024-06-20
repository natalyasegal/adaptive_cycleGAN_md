import argparse
import sys
import os
# Append the directory containing split.py to the path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from models.cycle_gan.adaptive_cycle_gan_pt import config_a_cgan, train_a_cgan

def main(args):
  config_a_cgan config
  config.num_epochs = args.epochs   
  config.texture_loss_weight = args.texture_loss_weight
  train_a_cgan()

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
    args = parser.parse_args()
    main(args)
