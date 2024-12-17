# src/models/base_model.py
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple

class BaseSteganoModel(nn.Module, ABC):
    """Base class for all steganography models"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.name = "base"
        
    @abstractmethod
    def encode(self, cover_image: torch.Tensor, secret_message: torch.Tensor) -> torch.Tensor:
        """Encode secret message into cover image"""
        pass
        
    @abstractmethod
    def decode(self, stego_image: torch.Tensor) -> torch.Tensor:
        """Decode secret message from stego image"""
        pass
        
    def get_metrics(self) -> Dict[str, float]:
        """Return model-specific metrics"""
        return {}

    @abstractmethod
    def get_model_size(self) -> int:
        """Return model size in parameters"""
        return sum(p.numel() for p in self.parameters())