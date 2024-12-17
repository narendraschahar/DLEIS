# src/data/transforms.py

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from typing import Dict, Optional

class SteganographyTransform:
    """Custom transform for steganography data"""
    
    def __init__(
        self,
        split: str,
        config: Optional[Dict] = None
    ):
        self.split = split
        self.config = config or {}
        
        # Get image size from config
        self.image_size = self.config.get('data', {}).get('image_size', 256)
        
        # Define transforms based on split
        if split == 'train':
            self.transform = T.Compose([
                T.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0)
                ),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = T.Compose([
                T.Resize(self.image_size),
                T.CenterCrop(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, img):
        return self.transform(img)

def get_transform(
    split: str,
    config: Optional[Dict] = None
) -> SteganographyTransform:
    """Get appropriate transform based on split and config"""
    return SteganographyTransform(split, config)

class CustomRandomCrop:
    """Custom random crop that maintains aspect ratio"""
    
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size
        
        if w == tw and h == th:
            return img
        
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        return TF.crop(img, i, j, th, tw)

class CustomColorJitter:
    """Custom color jitter with configurable parameters"""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, img):
        return self.color_jitter(img)

class RandomNoise:
    """Add random noise to image"""
    
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)

def create_validation_transform(
    image_size: int = 256
) -> T.Compose:
    """Create transform for validation/testing"""
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def create_train_transform(
    image_size: int = 256,
    augment: bool = True
) -> T.Compose:
    """Create transform for training"""
    transforms = [
        T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip()
    ]
    
    if augment:
        transforms.extend([
            CustomColorJitter(),
            T.RandomRotation(15),
            RandomNoise(0.01)
        ])
    
    transforms.extend([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transforms)