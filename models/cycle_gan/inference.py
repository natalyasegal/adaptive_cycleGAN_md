import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from models.cycle_gan.adaptive_cycle_gan_pt import Generator


# Function to preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)
    return image

# Function to postprocess the tensor to image
def postprocess_image(tensor, transform):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2  # Denormalize
    return transforms.ToPILImage()(tensor)

# Shifts distribution for  single image
def shift_one(test_image_path, device, transform, base_out_path = "out", show = True, save = True):
    # Load the trained models
    netG_A2B = Generator(input_nc=1, output_nc=1).to(device)
    netG_B2A = Generator(input_nc=1, output_nc=1).to(device)

    # Assuming the models are saved as 'netG_A2B.pth' and 'netG_B2A.pth'
    netG_A2B.load_state_dict(torch.load('netG_A2B.pth'))
    netG_B2A.load_state_dict(torch.load('netG_B2A.pth'))

    # Set models to evaluation mode
    netG_A2B.eval()
    netG_B2A.eval()

    # Load and preprocess the test image
    input_image = preprocess_image(test_image_path, transform).to(device)

    # Generate the translated image
    with torch.no_grad():
        translated_image = netG_A2B(input_image)

    # Postprocess and save the output image
    output_image = postprocess_image(translated_image, transform)

    if save:
      s = str(test_image_path.split('/')[-1])
      d = str(test_image_path.split('/')[-2])
      out_path  = f'{base_out_path}/{d}/{s}'
      output_image..convert('L').save(out_path)

    if show:
      # Display the output image
      plt.imshow(output_image, cmap='gray')
      plt.axis('off')
      plt.show()
    return output_image

def get_all_image_paths(root_dir):
    # Get a list of all image files in the directory and its subdirectories
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff']  # Add more extensions if needed
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(root_dir, '**', f'*.{ext}'), recursive=True))
    return image_paths

def batch_process_images(root_dir, base_out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the transformation to match the training phase
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for 1 channel
    ])

    # Get all image paths
    image_paths = get_all_image_paths(root_dir)
    for image_path in image_paths:
        try:
            shift_one(image_path, device, transform, base_out_path, show = False, save = True) 
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

    # Shifting of an example single image:
    test_image_path = './datasets/ed/A_from/class_1_blast/101.png'   
    image = shift_one(test_image_path, device, transform, show = False, save = False)
    image.save('sample_image_101.jpg')  

if __name__ == "__main__":
    root_dir = './datasets/ed/A_from/'  # Replace with the path to your directory
    base_out_path = "out"
    batch_process_images(root_dir, base_out_path)

