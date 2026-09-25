import torchvision.transforms.functional as F
import torch

def get_grayscale(image_tensor):
    return F.rgb_to_grayscale(image_tensor, num_output_channels=3)

def get_hue_rotation(image_tensor, hue_factor=0.3):
    return F.adjust_hue(image_tensor, hue_factor)