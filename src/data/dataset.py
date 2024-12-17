# src/data/dataset.py

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, Optional, Tuple
from .transforms import get_transform

class SteganoDataset(Dataset):
    """Base dataset class for steganography"""
    
    def __init__(
        self,
        data_dir: str,
        dataset_type: str,
        split: str,
        config: Optional[Dict] = None
    ):
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.config = config or {}
        
        # Get dataset-specific config
        dataset_config = self.config['data']['datasets'][split][dataset_type]
        self.transform = get_transform(split, self.config)
        
        # Setup paths
        self.image_paths = sorted(
            list(self.data_dir.glob('*.jpg')) +
            list(self.data_dir.glob('*.png')) +
            list(self.data_dir.glob('*.JPEG'))
        )
        
        # Verify dataset size
        expected_size = dataset_config['num_images']
        if len(self.image_paths) < expected_size:
            raise ValueError(
                f"Found only {len(self.image_paths)} images in {self.data_dir}, "
                f"expected {expected_size}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_dataloader(config: Dict, dataset_type: str, split: str) -> DataLoader:
    """Create dataloader for specified dataset and split"""
    
    dataset_config = config['data']['datasets'][split][dataset_type]
    data_dir = Path(config['data']['base_dir']) / dataset_config['path']
    
    dataset = SteganoDataset(
        data_dir=data_dir,
        dataset_type=dataset_type,
        split=split,
        config=config
    )
    
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(split == 'train'),
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

def create_dataloaders(config: Dict) -> Dict[str, DataLoader]:
    """Create all required dataloaders"""
    
    loaders = {}
    
    # Training loaders
    loaders['train_div2k'] = get_dataloader(config, 'div2k', 'train')
    loaders['train_coco'] = get_dataloader(config, 'coco', 'train')
    loaders['val'] = get_dataloader(config, 'imagenet', 'train')
    
    # Testing loaders
    loaders['test_mit'] = get_dataloader(config, 'mit_places', 'test')
    loaders['test_sipi'] = get_dataloader(config, 'usc_sipi', 'test')
    loaders['test_custom'] = get_dataloader(config, 'custom', 'test')
    
    return loaders
```