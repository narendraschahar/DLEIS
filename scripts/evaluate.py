# scripts/evaluate.py

import torch
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import ResultsVisualizer
from src.utils.attacks import SteganoAttacks

def evaluate_model(model, test_loader, config):
    """Evaluate a single model"""
    model.eval()
    metrics_calc = MetricsCalculator()
    attacks = SteganoAttacks()
    
    results = {
        'psnr': [],
        'ssim': [],
        'bpp': [],
        'message_mse': [],
        'robustness': {
            'jpeg': [],
            'noise': [],
            'rotation': [],
            'scaling': []
        }
    }
    
    with torch.no_grad():
        for cover, secret in tqdm(test_loader, desc=f"Evaluating {model.name}"):
            cover = cover.to(config['device'])
            secret = secret.to(config['device'])
            
            # Generate stego image
            stego = model.encode(cover, secret)
            recovered = model.decode(stego)
            
            # Calculate basic metrics
            metrics = metrics_calc.calculate_all_metrics(
                cover, stego, secret, recovered
            )
            
            for key in ['psnr', 'ssim', 'bpp', 'message_mse']:
                results[key].append(metrics[key])
            
            # Test robustness
            for attack_name, attack_fn in attacks.get_attacks().items():
                attacked_stego = attack_fn(stego)
                recovered_attacked = model.decode(attacked_stego)
                recovery_rate = metrics_calc.calculate_message_recovery(
                    secret, recovered_attacked
                )
                results['robustness'][attack_name].append(recovery_rate)
    
    # Average results
    final_results = {
        key: sum(values) / len(values)
        for key, values in results.items()
        if isinstance(values, list)
    }
    
    final_results['robustness'] = {
        key: sum(values) / len(values)
        for key, values in results['robustness'].items()
    }
    
    return final_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize models
    models = [
        RDN(config), ViTAA(config), PGN(config),
        DSA(config), WHN(config), MAT(config),
        EAPT(config), IPSN(config)
    ]
    
    # Load test data
    test_loader = get_test_loader(config)
    
    # Evaluate each model
    results = {}
    for model in models:
        print(f"\nEvaluating {model.name}...")
        results[model.name] = evaluate_model(model, test_loader, config)
    
    # Save results
    save_results(results, config)
    
    # Generate visualizations
    visualizer = ResultsVisualizer()
    visualizer.plot_metrics_comparison(results)
    visualizer.plot_robustness_comparison(results)
    
    # Print summary
    print("\nResults Summary:")
    print("=" * 50)
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        print(f"PSNR: {model_results['psnr']:.2f} dB")
        print(f"SSIM: {model_results['ssim']:.4f}")
        print("\nRobustness Results:")
        for attack, rate in model_results['robustness'].items():
            print(f"  {attack}: {rate:.4f}")

if __name__ == "__main__":
    main()