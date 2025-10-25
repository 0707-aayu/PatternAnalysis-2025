import argparse
import logging
import os
import time
import shutil

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
    # Helper to shape inputs into [1, 1, H, W]
    def prepare_input(arr):
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                arr = arr[np.newaxis, np.newaxis, :, :]
            elif arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
                arr = arr[np.newaxis, :, :, :]
        elif isinstance(arr, torch.Tensor):
            if arr.ndim == 2:
                arr = arr.unsqueeze(0).unsqueeze(0)
            elif arr.ndim == 3:
                arr = arr.permute(2, 0, 1).unsqueeze(0)
        return arr

    train_orig = prepare_input(train_orig)
    train_recon = prepare_input(train_recon)
    val_orig = prepare_input(val_orig)
    val_recon = prepare_input(val_recon)

    train_img = combine_images(train_orig, train_recon)
    val_img = combine_images(val_orig, val_recon)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(train_img, cmap='gray')
    plt.title("Train Original and Reconstructed")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(val_img, cmap='gray')
    plt.title("Validation Original and Reconstructed")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, f'epoch_{epoch}.png'))
    plt.close()

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

    for epoch in range(1, num_epochs + 1):
        train_commitment_loss, train_recon_loss, train_loss, train_ssim = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, num_epochs)
        val_commitment_loss, val_recon_loss, val_loss, val_ssim = validate_one_epoch(
            model, val_loader, criterion, device, epoch, num_epochs)

        if epoch % 5 == 0:
            save_epoch_image(
                train_loader.dataset[0].cpu().numpy(),
                model(train_loader.dataset[0].unsqueeze(0).to(device).float())[0].squeeze(0).cpu().detach().numpy(),
                val_loader.dataset[0].cpu().numpy(),
                model(val_loader.dataset[0].unsqueeze(0).to(device).float())[0].squeeze(0).cpu().detach().numpy(),
                epoch,
                image_dir)

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train SSIM: {train_ssim:.4f}, Val SSIM: {val_ssim:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

   


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQVAE on HipMRI dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    args = parser.parse_args()

    config = read_yaml_file(args.config) 
    train(config)

