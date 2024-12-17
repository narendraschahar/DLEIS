# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import numpy as np

class ResultsVisualizer:
    """Visualization utilities for steganography results"""
    
    @staticmethod
    def plot_metrics_comparison(
        results: Dict[str, Dict[str, float]],
        metric_names: List[str] = ['psnr', 'ssim'],
        save_path: str = None
    ):
        """Plot comparison of metrics across models"""
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        data = []
        for model_name, metrics in results.items():
            for metric in metric_names:
                data.append({
                    'Model': model_name,
                    'Metric': metric.upper(),
                    'Value': metrics[metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create plot
        sns.barplot(x='Model', y='Value', hue='Metric', data=df)
        plt.title('Model Performance Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_robustness_comparison(
        results: Dict[str, Dict[str, float]],
        save_path: str = None
    ):
        """Plot robustness comparison across models"""
        attacks = ['jpeg', 'noise', 'rotation', 'scaling']
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        models = list(results.keys())
        attack_values = {attack: [results[model]['robustness'][attack] 
                                for model in models] 
                        for attack in attacks}
        
        # Plot
        x = np.arange(len(models))
        width = 0.2
        multiplier = 0
        
        for attribute, measurement in attack_values.items():
            offset = width * multiplier
            plt.bar(x + offset, measurement, width, label=attribute.capitalize())
            multiplier += 1
        
        plt.xlabel('Models')
        plt.ylabel('Recovery Rate')
        plt.title('Robustness Against Different Attacks')
        plt.xticks(x + width, models, rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()