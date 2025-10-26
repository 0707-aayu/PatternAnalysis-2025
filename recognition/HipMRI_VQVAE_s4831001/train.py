import argparse
import logging
import os
import time
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader
from utils import get_transforms

from modules import VQVAE
from utils import calc_ssim, read_yaml_file, combine_images


def calculate_batch_ssim(batch: torch.Tensor, reconstructed_batch: torch.Tensor) -> float:
    batch_ssim = 0.0
    for i in range(batch.size(0)):
        original_image = batch[i, 0].cpu().detach().numpy()
        reconstructed_image = reconstructed_batch[i, 0].cpu().detach().numpy()
        batch_ssim += calc_ssim(original_image, reconstructed_image)
    return batch_ssim / batch.size(0)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_commitment_loss = 0.0
    total_recon_loss = 0.0
    total_loss = 0.0
    total_ssim = 0.0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}/{num_epochs}"):
        batch = batch.to(device).float()
        optimizer.zero_grad()
        reconstructed, commitment_loss = model(batch)
        recon_loss = criterion(reconstructed, batch)
        loss = recon_loss + commitment_loss
        loss.backward()
        optimizer.step()

        total_commitment_loss += commitment_loss.item()
        total_recon_loss += recon_loss.item()
        total_loss += loss.item()
        total_ssim += calculate_batch_ssim(batch, reconstructed)

    avg_commitment_loss = total_commitment_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    avg_ssim = total_ssim / len(train_loader)

    return avg_commitment_loss, avg_recon_loss, avg_total_loss, avg_ssim


def validate_one_epoch(model, val_loader, criterion, device, epoch, num_epochs):
    model.eval()
    total_commitment_loss = 0.0
    total_recon_loss = 0.0
    total_loss = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}/{num_epochs}"):
            batch = batch.to(device).float()
            reconstructed, commitment_loss = model(batch)
            recon_loss = criterion(reconstructed, batch)
            loss = recon_loss + commitment_loss

            total_commitment_loss += commitment_loss.item()
            total_recon_loss += recon_loss.item()
            total_loss += loss.item()
            total_ssim += calculate_batch_ssim(batch, reconstructed)

    avg_commitment_loss = total_commitment_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_total_loss = total_loss / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    return avg_commitment_loss, avg_recon_loss, avg_total_loss, avg_ssim


def save_epoch_image(train_orig, train_recon, val_orig, val_recon, epoch, image_dir):
     """
    Save a readable preview:
      Left panel  : [Train Original | Train Recon]
      Right panel : [Val   Original | Val   Recon]
    """    
    # ------ helpers ------
    def to_bchw(arr):
        """Ensure array is [1,1,H,W] or [B,1,H,W]. Accepts numpy or torch."""
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        # numpy path
        if arr.ndim == 2:                 # [H,W]
            arr = arr[None, None, ...]
        elif arr.ndim == 3:
            # could be [1,H,W] or [C,H,W] or [H,W,C]
            if arr.shape[0] in (1, 3):    # [C,H,W]
                if arr.shape[0] != 1:
                    arr = arr[:1]         # keep first channel
                arr = arr[None, ...]      # -> [1,1,H,W]
            elif arr.shape[-1] in (1, 3): # [H,W,C]
                arr = np.transpose(arr, (2, 0, 1))[None, ...]
                if arr.shape[1] != 1:
                    arr = arr[:, :1]
            else:                          # [1,H,W] already
                arr = arr[None, ...]
        # if [B,1,H,W] already, leave as is
        return arr

    def norm01(x):
        x = x.astype(np.float32)
        mn, mx = x.min(), x.max()
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x)
        return x

    # ------ standardize shapes ------
    train_orig = to_bchw(train_orig)
    train_recon = to_bchw(train_recon)
    val_orig   = to_bchw(val_orig)
    val_recon  = to_bchw(val_recon)

    # take the first image in the (mini)batch and squeeze to [H,W]
    to_img = lambda a: a[0, 0]
    tr_o = to_img(train_orig)
    tr_r = to_img(train_recon)
    va_o = to_img(val_orig)
    va_r = to_img(val_recon)

    # normalize each image to 0–1 for display
    tr_o, tr_r, va_o, va_r = map(norm01, (tr_o, tr_r, va_o, va_r))

    # build horizontal pairs: [original | reconstruction]
    train_pair = np.hstack([tr_o, tr_r])
    val_pair   = np.hstack([va_o, va_r])

    # ------ plot ------
    plt.figure(figsize=(10, 4.2))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(train_pair, cmap='gray', aspect='auto')
    ax1.set_title("Train Original and Reconstructed Image")
    ax1.axis('off')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(val_pair, cmap='gray', aspect='auto')
    ax2.set_title("Validation Original and Reconstructed Image")
    ax2.axis('off')

    plt.tight_layout(pad=1.2)
    os.makedirs(image_dir, exist_ok=True)
    plt.savefig(os.path.join(image_dir, f'epoch_{epoch}.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_training_curves(metrics, save_path):
    """
    Plot and save training/validation loss and SSIM curves
    """
    epochs = metrics['epochs']
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    train_ssim = metrics['train_ssim']
    val_ssim = metrics['val_ssim']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='s', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # SSIM curves
    axes[1].plot(epochs, train_ssim, label='Train SSIM', marker='o', linewidth=2, color='green')
    axes[1].plot(epochs, val_ssim, label='Val SSIM', marker='s', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_title('Training and Validation SSIM', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def train(config):
    # Setup variables
    model_params = config['model_parameters']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    weight_decay = config.get('weight_decay', 0)

    train_dir = config['train_dataset_dir']
    val_dir = config['val_dataset_dir']
    test_dir = config['test_dataset_dir']

    train_num_samples = config.get('train_num_samples', None)
    val_num_samples = config.get('val_num_samples', None)
    test_num_samples = config.get('test_num_samples', None)

    train_transforms = get_transforms(config.get('train_transforms', []))
    val_test_transforms = get_transforms(config.get('val_test_transforms', []))

    log_dir = config.get('log_dir', 'logs')
    image_dir = os.path.join(log_dir, 'images')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, optimizer
    model = VQVAE(**model_params).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_loader = get_dataloader(train_dir, batch_size, train_transforms, train_num_samples, shuffle=True)
    val_loader = get_dataloader(val_dir, batch_size, val_test_transforms, val_num_samples, shuffle=False)
    test_loader = get_dataloader(test_dir, 1, val_test_transforms, test_num_samples, shuffle=False)

    best_val_loss = float('inf')

    # Initialize metrics tracking
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_ssim': [],
        'val_ssim': []
    }

    for epoch in range(1, num_epochs + 1):
        train_commitment_loss, train_recon_loss, train_loss, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_commitment_loss, val_recon_loss, val_loss, val_ssim = validate_one_epoch(
            model, val_loader, criterion, device, epoch, num_epochs)
        
        # Track metrics
        metrics['epochs'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_ssim'].append(train_ssim)
        metrics['val_ssim'].append(val_ssim)

        # Save example images every 5 epochs
        if epoch % 5 == 0:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                tr_np = train_loader.dataset[0].cpu().numpy()  # [1,H,W]
                va_np = val_loader.dataset[0].cpu().numpy()    # [1,H,W]

                tr_rec = model(torch.from_numpy(tr_np).unsqueeze(0).to(device).float())[0].squeeze(0).cpu().numpy()
                va_rec = model(torch.from_numpy(va_np).unsqueeze(0).to(device).float())[0].squeeze(0).cpu().numpy()

                save_epoch_image(
                    tr_np,
                    tr_rec,
                    va_np,
                    va_rec,
                    epoch,
                    image_dir
                )
            if was_training:
                model.train()

            # Plot training curves
            plot_training_curves(metrics, os.path.join(log_dir, 'training_curves.png'))

        print(f"Epoch {epoch}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))


    # Save final metrics
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    plot_training_curves(metrics, os.path.join(log_dir, 'final_training_curves.png'))
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQVAE on HipMRI dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    config = read_yaml_file(args.config) 
    train(config)


