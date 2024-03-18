from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from timm.data.dataset import ImageDataset


# from matplotlib import pyplot as plt


def custom_transform(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = transform(image)
    return img


def load_image(path):
    # image_path = 'cello.jpg'
    image = Image.open(path)
    return image


def load_dataset(path):
    image_dataset = ImageDataset(path)
    return image_dataset


if __name__ == '__main__':
    dataset = ImageDataset('../image/')
    print(dataset[0])

    # # plot image
    # dataset[0][0].show()
    for img in dataset:
        img[0].show()
