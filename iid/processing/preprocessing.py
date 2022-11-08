from typing import List

import torch
import torchvision
from PIL import Image

image_to_tensor = torchvision.transforms.ToTensor()


def read_images(image_locations: List[str]) -> List[Image.Image]:
    """
    Reads and opens images from given locations using PIL.
    """
    return [Image.open(image) for image in image_locations]


def convert_images_to_tensors(
    images: List[Image.Image],
) -> List[torch.Tensor]:
    """
    Converts images to tensors using torchvision.transforms.
    """
    return [image_to_tensor(image) for image in images]
