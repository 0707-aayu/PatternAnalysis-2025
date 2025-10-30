# HipMRI VQ-VAE — Model inference and evaluation
# Author: Aayushi Arvind Dhande (s4831001)
# Description: Loads a trained VQ-VAE model and performs inference on the test set.
#              Computes SSIM/loss statistics and generates visual comparisons.

"""Inference and evaluation script for the HipMRI VQ-VAE model.

This script loads a trained VQVAE checkpoint, runs reconstruction on the test dataset,
computes pixel-wise MSE and SSIM metrics, and visualizes results with histogram plots
and side-by-side image comparisons.

Functions:
    - save_combined_image(): Save a combined view of original and reconstructed images.
    - predict(): Run model inference and generate SSIM/loss statistics and plots.

Usage:
    python predict.py --config config.yaml
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloader
from modules import VQVAE
from utils import calc_ssim, read_yaml_file, combine_images

def save_combined_image(original, reconstructed, save_path, title=None):
    """Save a combined visualization of the original and reconstructed image.

    Args:
        original (np.ndarray): 2D array of the original input image [H, W].
        reconstructed (np.ndarray): 2D array of the reconstructed image [H, W].
        save_path (str): File path to save the combined image.
        title (str | None): Optional title showing SSIM/Loss info.
    """
    # Reshape to [1, 1, H, W] for consistency with combine_images()
    original = original[None, None, :, :]
    reconstructed = reconstructed[None, None, :, :]
    combined = combine_images(original, reconstructed, max_images=1)
    plt.imshow(combined, cmap='gray')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def predict(config):
    """Run VQ-VAE inference on the test dataset and compute evaluation metrics.

    Args:
        config (dict): Configuration dictionary loaded from YAML, containing:
            - model_parameters: Hyperparameters for model instantiation.
            - pretrained_path: Path to the trained model weights (.pth).
            - test_dataset_dir: Directory containing test images.
            - save_dir (optional): Directory to save output predictions and plots.

    Workflow:
        1. Load the trained model and prepare test DataLoader.
        2. Perform reconstruction for all test samples.
        3. Compute MSE and SSIM metrics per sample.
        4. Save side-by-side comparison images.
        5. Aggregate and plot metric distributions.
    """
    # Select compute device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Read key configuration fields
    model_params = config['model_parameters']
    pretrained_path = config['pretrained_path']
    test_dataset_dir = config['test_dataset_dir']
    save_dir = config.get('save_dir', 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    
    # Load test data     
    test_loader = get_dataloader(test_dataset_dir, batch_size=1, shuffle=False)

    # Load trained model
    model = VQVAE(**model_params).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    criterion = nn.MSELoss()

    ssim_scores = []
    loss_scores = []

    # Disable gradients during inference for performance
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting"), 1):
            batch = batch.to(device).float()

            # Forward pass: reconstruction + codebook loss
            reconstructed, commitment_loss, _ = model(batch)
            recon_loss = criterion(reconstructed, batch)
            loss = recon_loss + commitment_loss

            # Compute SSIM between original and reconstructed image
            original_image = batch[0, 0].cpu().numpy()
            reconstructed_image = reconstructed[0, 0].cpu().numpy()
            ssim_value = calc_ssim(original_image, reconstructed_image)

            ssim_scores.append(ssim_value)
            loss_scores.append(loss.item())

            # Save a visual comparison for each test sample
            save_combined_image(
                original_image, reconstructed_image,
                os.path.join(save_dir, f'image_{i:04d}.png'),
                title=f"Loss: {loss.item():.4f}, SSIM: {ssim_value:.4f}",
            )
            
    # Convert lists to arrays for statistics         
    ssim_scores = np.array(ssim_scores)
    loss_scores = np.array(loss_scores)


    # --- Display & summarize test results ---
    print(f"\n=== Test Set SSIM Statistics ===")
    print(f"Mean SSIM: {ssim_scores.mean():.4f}")
    print(f"Median SSIM: {np.median(ssim_scores):.4f}")
    print(f"Min SSIM: {ssim_scores.min():.4f}")
    print(f"Max SSIM: {ssim_scores.max():.4f}")
    print(f"Std Dev SSIM: {ssim_scores.std():.4f}")

    print(f"\n=== Test Set Loss Statistics ===")
    print(f"Mean Loss: {loss_scores.mean():.4f}")
    print(f"Median Loss: {np.median(loss_scores):.4f}")
    print(f"Min Loss: {loss_scores.min():.4f}")
    print(f"Max Loss: {loss_scores.max():.4f}")

    # --- Generate histogram plots for SSIM and Loss ---
    plt.figure(figsize=(12, 5))

    # SSIM histogram
    plt.subplot(1, 2, 1)
    plt.hist(ssim_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('SSIM Score')
    plt.ylabel('Frequency')
    plt.title('SSIM Distribution on Test Set')
    plt.axvline(ssim_scores.mean(), color='red', linestyle='--', label=f'Mean: {ssim_scores.mean():.3f}')
    plt.axvline(np.median(ssim_scores), color='green', linestyle='--', label=f'Median: {np.median(ssim_scores):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss histogram
    plt.subplot(1, 2, 2)
    plt.hist(loss_scores, bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution on Test Set')
    plt.axvline(loss_scores.mean(), color='red', linestyle='--', label=f'Mean: {loss_scores.mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssim_loss_distribution.png'), dpi=300)
    plt.close()
    print(f"\nSSIM/Loss distribution plot saved to {save_dir}/ssim_loss_distribution.png")

# --- Entry point for CLI execution ---
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using VQVAE model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    
    config = read_yaml_file(args.config)
    predict(config)





