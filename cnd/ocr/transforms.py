import torch
import numpy as np
import cv2
import random
from albumentations import RandomBrightnessContrast
from albumentations import RandomContrast
from torchvision import transforms as T


class ImageScale(object):
    def __init__(self, size = (32, 80)):
        self.size = size

    def __call__(self, image):
        return cv2.resize(image, self.size)


class ToType(object):
    def __call__(self, image):
        return image.astype(np.uint8)


class ImageNormalization(object):
    def __call__(self, image):
        return image / 255.0


class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)[None, :, :]
        is_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if is_cuda else torch.device('cpu')
        return torch.from_numpy(image).to(device)


class RandomCrop(object):
    def __call__(self, image, width=32, height=80):
        assert image.shape[0] >= height, 'не корректная высота'
        assert image.shape[1] >= width, 'не корректная ширина'
        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)
        image = image[y:y+height, x:x+width]
        return image


class RandomFlip(object):
    def __call__(self, image):
        flip_modes = [-1, 0, 1, 'no flip']
        mode = random.choice(flip_modes)
        if mode != 'no flip':
            return cv2.flip(image, mode)
        return image


class BrightnessContrast(object):
    def __call__(self, image):
        return RandomBrightnessContrast()(image=image)['image']


class GrayTransform(object):
    def __call__(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

class ResizeToTensor(object):
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, image):
        image = image.astype('float32') / 255
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        image = image.astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)


def get_transforms(image_size):
    transform = T.Compose([
        GrayTransform(),
        ImageScale(image_size),
        ToType(),
        ImageNormalization(),
        BrightnessContrast(),
        ToTensor(),
    ])
    return transform
