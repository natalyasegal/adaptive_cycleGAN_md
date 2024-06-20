import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os
from efficientnet_pytorch import EfficientNet
import numpy as np

# Define CycleGAN model architectures (simplified version)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 7, stride=1, padding=3),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        model += [nn.Conv2d(64, output_nc, 7, stride=1, padding=3),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]
        model += [nn.Conv2d(512, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class EfficientNetTextureLoss(nn.Module):
    def __init__(self):
        super(EfficientNetTextureLoss, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.efficientnet.eval()
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_rgb = input.repeat(1, 3, 1, 1)  # Repeat grayscale image across 3 channels
        target_rgb = target.repeat(1, 3, 1, 1)  # Repeat grayscale image across 3 channels
        input_features = self.efficientnet.extract_features(input_rgb)
        target_features = self.efficientnet.extract_features(target_rgb)
        return nn.functional.mse_loss(input_features, target_features)

# need revisiting
def set_deterministic_settings():
    torch.manual_seed(11817572722858885327)
    np.random.seed(2147483648) #seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class config_a_cgan():
    def __init__(self):
        self.num_epochs = 20
        self.learning_rate = 0.0002
        self.beta1 = 0.5
        self.texture_loss_weight = 0.85  #0.9 #0.4  # Adjust this weight to put less emphasis on texture loss

def train_a_cgan(config, dataloader_A, dataloader_B):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training parameters
    num_epochs = config.num_epochs 
    learning_rate = config.learning_rate 
    beta1 = config.beta1
    texture_loss_weight = config.texture_loss_weight
    
    #set_deterministic_settings()
    #print(np.random.get_state()[1][0])
    #print(f'torch seed {torch.seed()}')
    
    # Initialize models
    netG_A2B = Generator(input_nc=1, output_nc=1).to(device)
    netG_B2A = Generator(input_nc=1, output_nc=1).to(device)
    netD_A = Discriminator(input_nc=1).to(device)
    netD_B = Discriminator(input_nc=1).to(device)
    
    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_texture = EfficientNetTextureLoss().to(device)

    # Optimizers
    optimizer_G = optim.Adam(list(netG_A2B.parameters()) + list(netG_B2A.parameters()), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        # Calculate texture loss weight - TODO: try this with different number of epochs and different ranges
        #texture_loss_weight = 1.0 - 0.6 * (epoch / num_epochs)  # Linearly decrease from 1 to 0.4
        texture_loss_weight = 1.0 - 0.9 * (epoch / num_epochs)  # Linearly decrease from 1 to 0.1
    
        for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_A = real_A[0].to(device)
            real_B = real_B[0].to(device)
    
            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), *netD_A(real_A).size()[1:]), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), *netD_A(real_A).size()[1:]), requires_grad=False).to(device)
    
            # ------------------
            #  Train Generators
                # ------------------
    
            # Identity loss
            loss_id_A = criterion_identity(netG_B2A(real_A), real_A)
            loss_id_B = criterion_identity(netG_A2B(real_B), real_B)
    
            # GAN loss
            fake_B = netG_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(netD_B(fake_B), valid)
            fake_A = netG_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(netD_A(fake_A), valid)
    
            # Cycle loss
            recov_A = netG_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = netG_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
    
            # Texture loss
            loss_texture_A = criterion_texture(fake_B, real_A)
            loss_texture_B = criterion_texture(fake_A, real_B)
    
            # Total loss
            loss_G = (loss_id_A + loss_id_B + loss_GAN_A2B + loss_GAN_B2A +
                      loss_cycle_A + loss_cycle_B +
                      texture_loss_weight * (loss_texture_A + loss_texture_B))
    
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
    
            # -----------------------
            #  Train Discriminators
            # -----------------------
    
            # Discriminator A
            loss_real_A = criterion_GAN(netD_A(real_A), valid)
            loss_fake_A = criterion_GAN(netD_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
    
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()
    
            # Discriminator B
            loss_real_B = criterion_GAN(netD_B(real_B), valid)
            loss_fake_B = criterion_GAN(netD_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
    
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()
    
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader_A)}] "
                      f"[D loss: {loss_D_A.item() + loss_D_B.item()}] "
                      f"[G loss: {loss_G.item()}]")
    
    # Save the trained models
    torch.save(netG_A2B.state_dict(), 'netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'netD_A.pth')
    torch.save(netD_B.state_dict(), 'netD_B.pth')
    
    print("Models saved successfully.")
