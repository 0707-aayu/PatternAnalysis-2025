# HipMRI VQ-VAE — Utility functions
# Author: Aayushi Arvind Dhande (s4831001)
# Description: Utility functions for data loading, augmentation, padding,
#              image combination, SSIM calculation, and YAML parsing.

"""Utility functions for the HipMRI VQ-VAE project.

This module provides supporting utilities for dataset loading and preprocessing,
visualization, image similarity evaluation, and configuration management.

Includes:
    - `load_data_2d()`: Load and normalize 2D slices from NIfTI files.
    - `pad_to_fixed_size()`: Zero-pad tensors to a common size.
    - `GaussianNoiseTransform`: Custom random noise augmentation.
    - `calc_ssim()`: Compute Structural Similarity Index between images.
    - `get_transforms()`: Build transform pipelines for augmentation.
    - `combine_images()`: Visualize original and reconstructed images.
    - `read_yaml_file()`: Parse YAML configuration files.
"""

import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision import transforms
import yaml

def pad_to_fixed_size(tensor, target_height, target_width):
    """Pad a tensor to the target height and width with zeros on the right/bottom.

    Args:
        tensor (torch.Tensor): Input tensor of shape [1, H, W].
        target_height (int): Desired height after padding.
        target_width (int): Desired width after padding.

    Returns:
        torch.Tensor: Zero-padded tensor of shape [1, target_height, target_width].
    """
    _, h, w = tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

class GaussianNoiseTransform:
    """Custom random Gaussian noise transform for data augmentation."""

    def __init__(self, std=0.05, p=0.5):
        """
        Args:
            std (float): Standard deviation of noise.
            p (float): Probability of applying noise.
        """
        self.std = std
        self.p = p
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise with probability `p`."""
        if torch.rand(1).item() < self.p:
            return x + torch.randn_like(x) * self.std
        return x

def load_data_2d(file_list, norm_image=True):
    """Load and normalize 2D slices from NIfTI files.

    Args:
        file_list (list): List of file paths to `.nii` or `.nii.gz` files.
        norm_image (bool): Whether to normalize intensities (zero mean, unit variance).

    Returns:
        list[torch.Tensor]: List of 2D tensors with shape [1, H, W].
    """
    temp_data = []
    max_height, max_width = 0, 0
    
    # --- First pass: load data and normalize ---
    for filepath in file_list:
        try:
            nii_img = nib.load(filepath)
            img = nii_img.get_fdata()

            # Handle 2D or 3D volumes
            if img.ndim == 2:
                slice_2d = img
                if norm_image:
                    slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)
                slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
                temp_data.append(slice_tensor)
            elif img.ndim == 3 and img.shape[2] == 1:
                slice_2d = img[:, :, 0]
                if norm_image:
                    slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)
                slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
                temp_data.append(slice_tensor)
            elif img.ndim == 3:
                for i in range(img.shape[2]):
                    slice_2d = img[:, :, i]
                    if norm_image:
                        slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)
                    slice_tensor = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
                    temp_data.append(slice_tensor)
            else:
                print(f"Skipping {filepath}: unexpected shape {img.shape}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    # --- Second pass: determine max dimensions ---
    for tensor in temp_data:
        _, h, w = tensor.shape
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w

    # --- Third pass: pad tensors to uniform size ---
    data = [pad_to_fixed_size(tensor, max_height, max_width) for tensor in temp_data]

    print(f"Total usable 2D slices loaded: {len(data)} with uniform size ({max_height}, {max_width})")
    return data



def calc_ssim(x, y):
    """Compute Structural Similarity Index (SSIM) between two 2D images.

    Args:
        x (np.ndarray): Image array [H, W] or [1, H, W].
        y (np.ndarray): Image array [H, W] or [1, H, W].

    Returns:
        float: SSIM score (1.0 = perfect similarity).
    """
    ssim_val = ssim(x.squeeze(), y.squeeze(), data_range=x.max() - x.min())
    return ssim_val



def get_transforms(transform_list):
    """Construct an augmentation pipeline compatible with tensor inputs [1, H, W].

    Args:
        transform_list (list[str]): List of transform names (e.g. 'RandomRotation', 'GaussianNoise').

    Returns:
        torchvision.transforms.Compose or None: Composed transform pipeline.
    """
    ops = []
    for t in (transform_list or []):
        if t == "RandomHorizontalFlip":
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        elif t == "RandomVerticalFlip":
            ops.append(transforms.RandomVerticalFlip(p=0.5))
        elif t == "RandomRotation":
            ops.append(transforms.RandomRotation(degrees=10))  # small, safe angles
        elif t == "RandomAffine":
            ops.append(transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ))
        elif t == "GaussianNoise":
            ops.append(GaussianNoiseTransform(std=0.05, p=0.5))
        # (leave unknown strings out silently)
    return transforms.Compose(ops) if ops else None
    
def read_yaml_file(filepath):
    """Load and parse a YAML configuration file.

    Args:
        filepath (str): Path to YAML file.

    Returns:
        dict: Parsed YAML contents as a dictionary.
    """
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data    

def combine_images(original_images, reconstructed_images, max_images=8):
    """
    Combine original and reconstructed images side-by-side in a numpy array for visualization.

    Args:
        original_images (torch.Tensor or np.ndarray): Batch of original images [B, 1, H, W]
        reconstructed_images (torch.Tensor or np.ndarray): Batch of reconstructed images [B, 1, H, W]
        max_images (int): Number of images to display from the batch

    Returns:
        np.ndarray: Combined image array with original and reconstructed images side by side.
    """
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.cpu().numpy()
    if isinstance(reconstructed_images, torch.Tensor):
        reconstructed_images = reconstructed_images.cpu().numpy()

    original_images = original_images[:max_images, 0, :, :]
    reconstructed_images = reconstructed_images[:max_images, 0, :, :]

    # Stack images vertically
    combined = np.vstack([
        np.hstack(original_images),
        np.hstack(reconstructed_images)
    ])

    # Robust normalization to avoid division by zero
    min_val = combined.min()
    max_val = combined.max()
    
    if max_val > min_val:
        combined = (combined - min_val) / (max_val - min_val)
    else:
        print(f"Warning: combined image has uniform value ({min_val}). Skipping normalization.")
        # Set to mid-gray for visibility
        combined = np.ones_like(combined) * 0.5

    return combined
