# scripts/generate_results.py

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import ResultsVisualizer
from src.utils.attacks import SteganoAttacks
from src.models import RDN, ViTAA, PGN, DSA, WHN, MAT, EAPT, IPSN

class ResultsGenerator:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_calc = MetricsCalculator()
        self.attacks = SteganoAttacks()
        
        # Initialize models
        self.models = self.initialize_models()
        
        # Set up paths for different result types
        self.figure_dir = self.output_dir / 'figures'
        self.table_dir = self.output_dir / 'tables'
        self.example_dir = self.output_dir / 'examples'
        
        for dir_path in [self.figure_dir, self.table_dir, self.example_dir]:
            dir_path.mkdir(exist_ok=True)

    def initialize_models(self):
        """Initialize all models with their configurations"""
        models = {
            'RDN': RDN(self.config['models']['rdn']),
            'ViT-AA': ViTAA(self.config['models']['vitaa']),
            'PGN': PGN(self.config['models']['pgn']),
            'DSA': DSA(self.config['models']['dsa']),
            'WHN': WHN(self.config['models']['whn']),
            'MAT': MAT(self.config['models']['mat']),
            'EAPT': EAPT(self.config['models']['eapt']),
            'IPSN': IPSN(self.config['models']['ipsn'])
        }
        
        # Load pretrained weights
        for name, model in models.items():
            weights_path = Path(self.config['weights_dir']) / f"{name.lower()}_weights.pth"
            if weights_path.exists():
                model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
        
        return models

    def generate_performance_table(self, results):
        """Generate LaTeX table for model performance metrics"""
        table = """
        \\begin{table}[htbp]
        \\centering
        \\begin{tabular}{lccccc}
        \\toprule
        Model & PSNR (dB) & SSIM & BPP & Time (ms) & Memory (GB) \\\\
        \\midrule
        """
        
        for model_name, metrics in results.items():
            row = f"{model_name} & {metrics['psnr']:.2f} & {metrics['ssim']:.4f} & "
            row += f"{metrics['bpp']:.2f} & {metrics['time']:.1f} & {metrics['memory']:.1f} \\\\\n"
            table += row
        
        table += """
        \\bottomrule
        \\end{tabular}
        \\caption{Performance comparison of steganography models}
        \\label{tab:performance}
        \\end{table}
        """
        
        with open(self.table_dir / 'performance.tex', 'w') as f:
            f.write(table)

    def generate_robustness_comparison(self, results):
        """Generate robustness comparison visualization"""
        plt.figure(figsize=(12, 6))
        
        data = []
        for model_name, metrics in results.items():
            for attack, value in metrics['robustness'].items():
                data.append({
                    'Model': model_name,
                    'Attack': attack.capitalize(),
                    'Recovery Rate': value
                })
        
        df = pd.DataFrame(data)
        sns.barplot(x='Model', y='Recovery Rate', hue='Attack', data=df)
        plt.title('Robustness Against Different Attacks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'robustness_comparison.png', dpi=300)
        plt.close()

    def generate_visual_examples(self, test_loader):
        """Generate visual examples of steganography results"""
        cover_images, secret_messages = next(iter(test_loader))
        cover_images = cover_images.to(self.device)
        secret_messages = secret_messages.to(self.device)
        
        fig, axes = plt.subplots(
            len(self.models),
            4,
            figsize=(16, 4 * len(self.models))
        )
        
        for i, (name, model) in enumerate(self.models.items()):
            with torch.no_grad():
                stego = model.encode(cover_images, secret_messages)
                recovered = model.decode(stego)
                
                # Calculate metrics
                psnr = self.metrics_calc.calculate_psnr(
                    cover_images[0],
                    stego[0]
                )
                ssim = self.metrics_calc.calculate_ssim(
                    cover_images[0],
                    stego[0]
                )
                
                # Plot results
                axes[i, 0].imshow(self.tensor_to_numpy(cover_images[0]))
                axes[i, 0].set_title('Cover Image')
                axes[i, 1].imshow(self.tensor_to_numpy(secret_messages[0]))
                axes[i, 1].set_title('Secret Message')
                axes[i, 2].imshow(self.tensor_to_numpy(stego[0]))
                axes[i, 2].set_title(f'Stego Image\nPSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}')
                axes[i, 3].imshow(self.tensor_to_numpy(recovered[0]))
                axes[i, 3].set_title('Recovered Message')
                
                axes[i, 0].set_ylabel(name)
                
                for ax in axes[i]:
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'visual_examples.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_attack_visualization(self, test_loader):
        """Generate visualization of different attacks and recovery"""
        cover_image, secret_message = next(iter(test_loader))
        cover_image = cover_image[0:1].to(self.device)
        secret_message = secret_message[0:1].to(self.device)
        
        attacks = {
            'JPEG': self.attacks.jpeg_compression,
            'Noise': self.attacks.add_noise,
            'Rotation': self.attacks.rotation,
            'Scaling': self.attacks.scaling
        }
        
        fig, axes = plt.subplots(
            len(self.models),
            len(attacks) + 2,
            figsize=(4 * (len(attacks) + 2), 4 * len(self.models))
        )
        
        for i, (model_name, model) in enumerate(self.models.items()):
            with torch.no_grad():
                stego = model.encode(cover_image, secret_message)
                
                # Original stego image
                axes[i, 0].imshow(self.tensor_to_numpy(stego[0]))
                axes[i, 0].set_title('Original Stego')
                
                # Apply each attack
                for j, (attack_name, attack_fn) in enumerate(attacks.items(), 1):
                    attacked = attack_fn(stego)
                    recovered = model.decode(attacked)
                    
                    axes[i, j].imshow(self.tensor_to_numpy(attacked[0]))
                    axes[i, j].set_title(f'{attack_name} Attack')
                    
                    # Show recovered message
                    if j == len(attacks):
                        axes[i, j+1].imshow(self.tensor_to_numpy(recovered[0]))
                        axes[i, j+1].set_title('Recovered')
                
                axes[i, 0].set_ylabel(model_name)
                
                for ax in axes[i]:
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.figure_dir / 'attack_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def tensor_to_numpy(tensor):
        """Convert tensor to numpy array for visualization"""
        return tensor.cpu().permute(1, 2, 0).numpy()

    def evaluate_models(self, test_loader):
        """Evaluate all models and collect results"""
        results = {}
        
        for name, model in tqdm(self.models.items(), desc="Evaluating models"):
            model_results = {
                'psnr': [],
                'ssim': [],
                'bpp': [],
                'time': [],
                'memory': [],
                'robustness': {
                    'jpeg': [],
                    'noise': [],
                    'rotation': [],
                    'scaling': []
                }
            }
            
            for cover, secret in tqdm(test_loader, desc=f"Testing {name}"):
                cover = cover.to(self.device)
                secret = secret.to(self.device)
                
                with torch.no_grad():
                    # Measure encoding time
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    stego = model.encode(cover, secret)
                    end_time.record()
                    torch.cuda.synchronize()
                    
                    model_results['time'].append(
                        start_time.elapsed_time(end_time)
                    )
                    
                    # Calculate basic metrics
                    metrics = self.metrics_calc.calculate_all_metrics(
                        cover, stego, secret, model.decode(stego)
                    )
                    
                    for key in ['psnr', 'ssim', 'bpp']:
                        model_results[key].append(metrics[key])
                    
                    # Test robustness
                    for attack_name, attack_fn in self.attacks.get_attacks().items():
                        attacked = attack_fn(stego)
                        recovered = model.decode(attacked)
                        recovery_rate = self.metrics_calc.calculate_message_recovery(
                            secret, recovered
                        )
                        model_results['robustness'][attack_name].append(recovery_rate)
            
            # Average results
            results[name] = {
                key: np.mean(values) if isinstance(values, list)
                else {k: np.mean(v) for k, v in values.items()}
                for key, values in model_results.items()
            }
            
            # Add memory usage
            results[name]['memory'] = torch.cuda.max_memory_allocated() / 1e9  # Convert to GB
        
        return results

    def generate_all_results(self, test_loader):
        """Generate all results and save them"""
        # Evaluate models
        results = self.evaluate_models(test_loader)
        
        # Generate tables
        self.generate_performance_table(results)
        
        # Generate visualizations
        self.generate_robustness_comparison(results)
        self.generate_visual_examples(test_loader)
        self.generate_attack_visualization(test_loader)
        
        # Save numerical results
        with open(self.output_dir / 'numerical_results.yaml', 'w') as f:
            yaml.dump(results, f)

def main():
    parser = argparse.ArgumentParser(description='Generate steganography results')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create results generator
    generator = ResultsGenerator(config, args.output_dir)
    
    # Get test data loader
    test_loader = get_test_loader(config)  # You'll need to implement this
    
    # Generate all results
    generator.generate_all_results(test_loader)

if __name__ == '__main__':
    main()