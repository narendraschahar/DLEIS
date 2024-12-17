# tests/test_models.py

import pytest
import torch
import numpy as np
from src.models import RDN, ViTAA, PGN, DSA, WHN, MAT, EAPT, IPSN
from src.utils.metrics import MetricsCalculator

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sample_data(device):
    cover = torch.randn(2, 3, 256, 256).to(device)
    secret = torch.randn(2, 3, 256, 256).to(device)
    return cover, secret

@pytest.fixture
def metrics_calc():
    return MetricsCalculator()

def test_model_output_shapes(sample_data, device):
    models = [
        RDN({'num_features': 64, 'growth_rate': 32, 'num_blocks': 16}),
        ViTAA({'patch_size': 16, 'embed_dim': 768, 'num_heads': 12}),
        PGN({'num_stages': 4, 'base_channels': 64}),
        DSA({'channels': 64, 'num_blocks': 8}),
        WHN({'num_blocks': 6, 'base_channels': 64}),
        MAT({'embed_dim': 256, 'num_heads': 8, 'num_layers': 6}),
        EAPT({'embed_dim': 256, 'num_pyramid_levels': 3}),
        IPSN({'channels': 64})
    ]
    
    cover, secret = sample_data
    
    for model in models:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Test encoding
            stego = model.encode(cover, secret)
            assert stego.shape == cover.shape, f"{model.name} encode shape mismatch"
            
            # Test decoding
            recovered = model.decode(stego)
            assert recovered.shape == secret.shape, f"{model.name} decode shape mismatch"

def test_model_metrics(sample_data, metrics_calc, device):
    models = [
        RDN({'num_features': 64, 'growth_rate': 32, 'num_blocks': 16}),
        ViTAA({'patch_size': 16, 'embed_dim': 768, 'num_heads': 12}),
        PGN({'num_stages': 4, 'base_channels': 64}),
        DSA({'channels': 64, 'num_blocks': 8}),
        WHN({'num_blocks': 6, 'base_channels': 64}),
        MAT({'embed_dim': 256, 'num_heads': 8, 'num_layers': 6}),
        EAPT({'embed_dim': 256, 'num_pyramid_levels': 3}),
        IPSN({'channels': 64})
    ]
    
    cover, secret = sample_data
    min_psnr = 30.0  # Minimum acceptable PSNR
    min_ssim = 0.90  # Minimum acceptable SSIM
    
    for model in models:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            stego = model.encode(cover, secret)
            recovered = model.decode(stego)
            
            metrics = metrics_calc.calculate_all_metrics(
                cover, stego, secret, recovered
            )
            
            assert metrics['psnr'] > min_psnr, \
                f"{model.name} PSNR too low: {metrics['psnr']}"
            assert metrics['ssim'] > min_ssim, \
                f"{model.name} SSIM too low: {metrics['ssim']}"

def test_model_robustness(sample_data, metrics_calc, device):
    models = [
        RDN({'num_features': 64, 'growth_rate': 32, 'num_blocks': 16}),
        ViTAA({'patch_size': 16, 'embed_dim': 768, 'num_heads': 12}),
        PGN({'num_stages': 4, 'base_channels': 64}),
        DSA({'channels': 64, 'num_blocks': 8}),
        WHN({'num_blocks': 6, 'base_channels': 64}),
        MAT({'embed_dim': 256, 'num_heads': 8, 'num_layers': 6}),
        EAPT({'embed_dim': 256, 'num_pyramid_levels': 3}),
        IPSN({'channels': 64})
    ]
    
    cover, secret = sample_data
    min_recovery_rate = 0.8  # Minimum acceptable recovery rate
    
    for model in models:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            stego = model.encode(cover, secret)
            
            # Test JPEG compression
            compressed = jpeg_compress(stego)
            recovered = model.decode(compressed)
            recovery_rate = metrics_calc.calculate_message_recovery(
                secret, recovered
            )
            assert recovery_rate > min_recovery_rate, \
                f"{model.name} JPEG recovery rate too low: {recovery_rate}"
            
            # Test noise addition
            noisy = add_noise(stego)
            recovered = model.decode(noisy)
            recovery_rate = metrics_calc.calculate_message_recovery(
                secret, recovered
            )
            assert recovery_rate > min_recovery_rate, \
                f"{model.name} noise recovery rate too low: {recovery_rate}"

# tests/test_models.py (continued)

def test_model_memory_efficiency():
    models = [
        RDN({'num_features': 64, 'growth_rate': 32, 'num_blocks': 16}),
        ViTAA({'patch_size': 16, 'embed_dim': 768, 'num_heads': 12}),
        PGN({'num_stages': 4, 'base_channels': 64}),
        DSA({'channels': 64, 'num_blocks': 8}),
        WHN({'num_blocks': 6, 'base_channels': 64}),
        MAT({'embed_dim': 256, 'num_heads': 8, 'num_layers': 6}),
        EAPT({'embed_dim': 256, 'num_pyramid_levels': 3}),
        IPSN({'channels': 64})
    ]
    
    max_params = 50_000_000  # 50M parameters
    max_memory = 10.0  # 10 GB
    
    for model in models:
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params < max_params, \
            f"{model.name} has too many parameters: {num_params}"
        
        # Estimate memory usage
        memory_usage = estimate_model_memory(model)
        assert memory_usage < max_memory, \
            f"{model.name} uses too much memory: {memory_usage}GB"

# tests/test_metrics.py

def test_metrics_calculation():
    metrics_calc = MetricsCalculator()
    
    # Create sample data
    cover = torch.rand(1, 3, 256, 256)
    stego = cover + 0.01 * torch.rand(1, 3, 256, 256)  # Slight modification
    secret = torch.rand(1, 3, 256, 256)
    recovered = secret + 0.01 * torch.rand(1, 3, 256, 256)
    
    metrics = metrics_calc.calculate_all_metrics(cover, stego, secret, recovered)
    
    # Test PSNR calculation
    assert metrics['psnr'] > 0, "PSNR should be positive"
    assert metrics['psnr'] < 100, "PSNR too high to be realistic"
    
    # Test SSIM calculation
    assert 0 <= metrics['ssim'] <= 1, "SSIM should be between 0 and 1"
    
    # Test BPP calculation
    assert metrics['bpp'] > 0, "BPP should be positive"

# scripts/generate_results.py

import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.visualization import ResultsVisualizer
from src.utils.metrics import MetricsCalculator

def generate_comparison_tables(results, output_path):
    """Generate LaTeX tables for paper"""
    # Model performance table
    performance_table = """
    \\begin{table}[h]
    \\centering
    \\begin{tabular}{lccccc}
    \\toprule
    Model & PSNR (dB) & SSIM & BPP & Time (ms) & Memory (GB) \\\\
    \\midrule
    """
    
    for model_name, metrics in results.items():
        performance_table += f"{model_name} & {metrics['psnr']:.2f} & {metrics['ssim']:.4f} & "
        performance_table += f"{metrics['bpp']:.2f} & {metrics['time']:.1f} & {metrics['memory']:.1f} \\\\\n"
    
    performance_table += """
    \\bottomrule
    \\end{tabular}
    \\caption{Performance comparison of steganography models}
    \\label{tab:performance}
    \\end{table}
    """
    
    # Robustness table
    robustness_table = """
    \\begin{table}[h]
    \\centering
    \\begin{tabular}{lcccc}
    \\toprule
    Model & JPEG (Q=75) & Noise & Rotation & Scaling \\\\
    \\midrule
    """
    
    for model_name, metrics in results.items():
        robustness = metrics['robustness']
        robustness_table += f"{model_name} & {robustness['jpeg']:.3f} & "
        robustness_table += f"{robustness['noise']:.3f} & {robustness['rotation']:.3f} & "
        robustness_table += f"{robustness['scaling']:.3f} \\\\\n"
    
    robustness_table += """
    \\bottomrule
    \\end{tabular}
    \\caption{Robustness comparison against different attacks}
    \\label{tab:robustness}
    \\end{table}
    """
    
    # Save tables
    with open(output_path / 'performance_table.tex', 'w') as f:
        f.write(performance_table)
    with open(output_path / 'robustness_table.tex', 'w') as f:
        f.write(robustness_table)

def generate_visual_examples(models, test_loader, output_path):
    """Generate visual examples for paper"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get a batch of test images
    cover_images, secret_messages = next(iter(test_loader))
    cover_images = cover_images.to(device)
    secret_messages = secret_messages.to(device)
    
    fig, axes = plt.subplots(len(models), 4, figsize=(16, 4*len(models)))
    
    for i, (name, model) in enumerate(models.items()):
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            stego = model.encode(cover_images, secret_messages)
            recovered = model.decode(stego)
            
            # Plot results
            axes[i, 0].imshow(to_numpy(cover_images[0]))
            axes[i, 0].set_title('Cover Image')
            axes[i, 1].imshow(to_numpy(secret_messages[0]))
            axes[i, 1].set_title('Secret Message')
            axes[i, 2].imshow(to_numpy(stego[0]))
            axes[i, 2].set_title('Stego Image')
            axes[i, 3].imshow(to_numpy(recovered[0]))
            axes[i, 3].set_title('Recovered Message')
            
            axes[i, 0].set_ylabel(name)
    
    plt.tight_layout()
    plt.savefig(output_path / 'visual_examples.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize models and get results
    models = initialize_models(config)
    results = evaluate_all_models(models, config)
    
    # Generate visualizations
    visualizer = ResultsVisualizer()
    visualizer.plot_metrics_comparison(results, save_path=output_path / 'metrics.png')
    visualizer.plot_robustness_comparison(results, save_path=output_path / 'robustness.png')
    
    # Generate LaTeX tables
    generate_comparison_tables(results, output_path)
    
    # Generate visual examples
    test_loader = get_test_loader(config)
    generate_visual_examples(models, test_loader, output_path)

if __name__ == '__main__':
    main()