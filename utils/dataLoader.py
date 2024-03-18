from PIL import Image
import numpy as np
import torchvision.transforms as transforms


def custom_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transform(image)
    return img


def load_images(path):
    # image_path = 'cello.jpg'
    image = Image.open(path)
    return image
