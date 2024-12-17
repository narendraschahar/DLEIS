# scripts/train.py

import torch
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.utils.metrics import MetricsCalculator
from src.losses.stego_losses import StegoLoss

def train_epoch(model, train_loader, criterion, optimizer, config):
    model.train()
    total_loss = 0
    metrics_calc = MetricsCalculator()
    
    with tqdm(train_loader, desc='Training') as pbar:
        for batch_idx, (cover, secret) in enumerate(pbar):
            cover = cover.to(config['device'])
            secret = secret.to(config['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            stego = model.encode(cover, secret)
            recovered = model.decode(stego)
            
            # Calculate loss
            losses = criterion(cover, stego, secret, recovered)
            total_loss = losses['total']
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Calculate metrics
            metrics = metrics_calc.calculate_all_metrics(
                cover, stego, secret, recovered
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim']
            })
            
    return total_loss.item(), metrics

def validate(model, val_loader, criterion, config):
    model.eval()
    total_loss = 0
    metrics_calc = MetricsCalculator()
    all_metrics = []
    
    with torch.no_grad():
        for cover, secret in val_loader:
            cover = cover.to(config['device'])
            secret = secret.to(config['device'])
            
            stego = model.encode(cover, secret)
            recovered = model.decode(stego)
            
            losses = criterion(cover, stego, secret, recovered)
            total_loss += losses['total'].item()
            
            metrics = metrics_calc.calculate_all_metrics(
                cover, stego, secret, recovered
            )
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: sum(d[k] for d in all_metrics) / len(all_metrics)
        for k in all_metrics[0].keys()
    }
    
    return total_loss / len(val_loader), avg_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup model and training
    model = get_model(args.model, config)
    criterion = StegoLoss(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Setup data loaders
    train_loader, val_loader = get_dataloaders(config)
    
    # Training loop
    writer = SummaryWriter(f"runs/{model.name}")
    best_psnr = 0
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, config
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, config)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for metric, value in val_metrics.items():
            writer.add_scalar(f'Metrics/{metric}', value, epoch)
        
        # Save best model
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            save_checkpoint(model, optimizer, epoch, val_metrics, config)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"PSNR: {val_metrics['psnr']:.2f}, "
              f"SSIM: {val_metrics['ssim']:.4f}")

if __name__ == '__main__':
    main()