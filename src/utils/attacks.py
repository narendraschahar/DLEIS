# src/utils/attacks.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Callable, Union, Tuple
from torchvision.transforms import functional as TF
import kornia.augmentation as K

class SteganoAttacks:
    """Implementation of various attacks for steganography robustness testing"""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Kornia augmentations
        self.jpeg = K.RandomJPEG(p=1.0)
        self.noise = K.RandomGaussianNoise(p=1.0)
        self.rotate = K.RandomRotation(degrees=5.0, p=1.0)
        
    def get_attacks(self) -> Dict[str, Callable]:
        """Return dictionary of all available attacks"""
        return {
            'jpeg': self.jpeg_compression,
            'noise': self.add_noise,
            'rotation': self.rotation,
            'scaling': self.scaling,
            'blur': self.gaussian_blur,
            'crop': self.random_crop,
            'combined': self.combined_attack
        }
    
    def jpeg_compression(
        self,
        image: torch.Tensor,
        quality: int = 75
    ) -> torch.Tensor:
        """Apply JPEG compression attack"""
        B, C, H, W = image.shape
        image = image.clamp(0, 1)  # Ensure valid range
        
        # Convert to numpy for OpenCV
        image_np = image.permute(0, 2, 3, 1).cpu().numpy() * 255
        compressed = []
        
        for i in range(B):
            # Encode and decode with JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, encoded = cv2.imencode('.jpg', image_np[i], encode_param)
            decoded = cv2.imdecode(encoded, 1)
            
            # Convert back to tensor
            compressed.append(torch.from_numpy(decoded).float() / 255.0)
        
        # Stack and reshape
        compressed = torch.stack(compressed)
        compressed = compressed.permute(0, 3, 1, 2).to(self.device)
        
        return compressed
    
    def add_noise(
        self,
        image: torch.Tensor,
        std: float = 0.1,
        noise_type: str = 'gaussian'
    ) -> torch.Tensor:
        """Add noise to image"""
        if noise_type == 'gaussian':
            noise = torch.randn_like(image) * std
        elif noise_type == 'salt_and_pepper':
            noise = torch.zeros_like(image)
            noise.bernoulli_(0.05)
            noise = noise * 2 - 1  # Convert to -1 and 1
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)
    
    def rotation(
        self,
        image: torch.Tensor,
        angle: float = 5.0,
        resample: int = TF.InterpolationMode.BILINEAR
    ) -> torch.Tensor:
        """Apply rotation attack"""
        return TF.rotate(image, angle, resample=resample)
    
    def scaling(
        self,
        image: torch.Tensor,
        scale_factor: float = 0.5
    ) -> torch.Tensor:
        """Apply scaling attack"""
        B, C, H, W = image.shape
        
        # Scale down
        scaled_h = int(H * scale_factor)
        scaled_w = int(W * scale_factor)
        scaled_down = F.interpolate(
            image,
            size=(scaled_h, scaled_w),
            mode='bilinear',
            align_corners=False
        )
        
        # Scale back up
        scaled_up = F.interpolate(
            scaled_down,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return scaled_up
    
    def gaussian_blur(
        self,
        image: torch.Tensor,
        kernel_size: int = 5,
        sigma: float = 1.0
    ) -> torch.Tensor:
        """Apply Gaussian blur attack"""
        return TF.gaussian_blur(image, kernel_size, sigma)
    
    def random_crop(
        self,
        image: torch.Tensor,
        crop_size: Union[int, Tuple[int, int]] = None,
        padding_mode: str = 'reflect'
    ) -> torch.Tensor:
        """Apply random crop and resize back attack"""
        B, C, H, W = image.shape
        
        if crop_size is None:
            crop_size = (int(H * 0.9), int(W * 0.9))
        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        # Random crop
        cropped = K.RandomCrop(crop_size, padding_mode=padding_mode)(image)
        
        # Resize back to original size
        resized = F.interpolate(
            cropped,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return resized
    
    def combined_attack(
        self,
        image: torch.Tensor,
        attack_params: Dict = None
    ) -> torch.Tensor:
        """Apply multiple attacks in sequence"""
        if attack_params is None:
            attack_params = {
                'jpeg_quality': 75,
                'noise_std': 0.05,
                'rotation_angle': 2.0,
                'scale_factor': 0.8
            }
        
        # Apply sequence of attacks
        attacked = image
        attacked = self.jpeg_compression(attacked, attack_params['jpeg_quality'])
        attacked = self.add_noise(attacked, attack_params['noise_std'])
        attacked = self.rotation(attacked, attack_params['rotation_angle'])
        attacked = self.scaling(attacked, attack_params['scale_factor'])
        
        return attacked
    
    @staticmethod
    def calculate_robustness(
        original: torch.Tensor,
        attacked: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate robustness metrics"""
        # Convert to numpy for calculations
        original_np = original.detach().cpu().numpy()
        attacked_np = attacked.detach().cpu().numpy()
        
        # Calculate metrics
        mse = np.mean((original_np - attacked_np) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        # Structural similarity
        ssim = 0.0
        for i in range(original.shape[0]):
            ssim += structural_similarity(
                original_np[i].transpose(1, 2, 0),
                attacked_np[i].transpose(1, 2, 0),
                multichannel=True
            )
        ssim /= original.shape[0]
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'ssim': float(ssim)
        }
    
    def test_robustness(
        self,
        model: torch.nn.Module,
        stego_image: torch.Tensor,
        secret_message: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Test model robustness against all attacks"""
        results = {}
        
        for attack_name, attack_fn in self.get_attacks().items():
            # Apply attack
            attacked_stego = attack_fn(stego_image)
            
            # Try to recover message
            recovered_message = model.decode(attacked_stego)
            
            # Calculate recovery metrics
            metrics = self.calculate_robustness(secret_message, recovered_message)
            results[attack_name] = metrics
        
        return results

# Example usage:
if __name__ == "__main__":
    # Create attacks instance
    attacks = SteganoAttacks()
    
    # Create sample image
    image = torch.rand(1, 3, 256, 256)
    
    # Apply different attacks
    jpeg_attacked = attacks.jpeg_compression(image)
    noisy_image = attacks.add_noise(image)
    rotated_image = attacks.rotation(image)
    scaled_image = attacks.scaling(image)
    
    # Combined attack
    combined = attacks.combined_attack(image)
    
    print("Attacks applied successfully!")