# src/utils/metrics.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, Tuple

class MetricsCalculator:
    """Calculate various steganography metrics"""
    
    @staticmethod
    def calculate_all_metrics(
        cover_image: torch.Tensor,
        stego_image: torch.Tensor,
        secret_message: torch.Tensor,
        recovered_message: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate all relevant metrics"""
        metrics = {}
        
        # Image quality metrics
        metrics['psnr'] = MetricsCalculator.calculate_psnr(
            cover_image.cpu().numpy(), 
            stego_image.cpu().numpy()
        )
        metrics['ssim'] = MetricsCalculator.calculate_ssim(
            cover_image.cpu().numpy(), 
            stego_image.cpu().numpy()
        )
        
        # Message recovery metrics
        metrics['message_mse'] = MetricsCalculator.calculate_message_mse(
            secret_message,
            recovered_message
        )
        
        # Capacity metrics
        metrics['bpp'] = MetricsCalculator.calculate_bpp(
            cover_image.shape[-2:],
            secret_message.shape[-2:]
        )
        
        return metrics

    @staticmethod
    def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        return psnr(original, modified)
    
    @staticmethod
    def calculate_ssim(original: np.ndarray, modified: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        return ssim(original, modified, multichannel=True)
    
    @staticmethod
    def calculate_message_mse(
        original_message: torch.Tensor,
        recovered_message: torch.Tensor
    ) -> float:
        """Calculate MSE between original and recovered messages"""
        return torch.mean((original_message - recovered_message) ** 2).item()

    @staticmethod
    def calculate_bpp(
        image_size: Tuple[int, int],
        message_size: Tuple[int, int]
    ) -> float:
        """Calculate Bits Per Pixel"""
        return (message_size[0] * message_size[1]) / (image_size[0] * image_size[1])