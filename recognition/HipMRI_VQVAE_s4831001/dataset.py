# HipMRI VQ-VAE — Dataset utilities
# Author: Aayushi Arvind Dhande (s4831001)
# Description: Load NIfTI slices, optionally normalize/transform them, and
#              expose a PyTorch Dataset plus a convenience DataLoader builder.

"""Dataset helpers for HipMRI 2D slice experiments.

Loads NIfTI (.nii) files via `utils.load_data_2d`, which returns
normalized/padded tensors shaped [1, H, W]. Provides:
- `HipMRIDataset`: indexable Dataset over 2D slices
- `get_dataloader(...)`: convenience function to build a DataLoader
"""

import os
from torch.utils.data import Dataset, DataLoader
import torch
from utils import load_data_2d

class HipMRIDataset(Dataset):
    """
    PyTorch Dataset for HipMRI processed 2D slices.

    Expects a list of file paths; `load_data_2d` loads volumes, extracts slices,
    and (optionally) z-score normalizes them. Each item is a tensor of shape [1, H, W].
    """

    def __init__(self, file_list, norm_image=True, transform=None):
        """Initialize the dataset.
        Args:
            file_list (list): Full paths to NIfTI files (e.g., .nii or .nii.gz).
            norm_image (bool): If True, z-score each slice during loading.
            transform (callable|None): Optional transform applied per-sample
                (expects and returns a tensor shaped [1, H, W]).
        """
        # Load once up front; this returns a list[Tensor] with consistent spatial size.
        self.data = load_data_2d(file_list, norm_image=norm_image)
        self.transform = transform

    def __len__(self):
        """Number of 2D slices available."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return one slice (tensor [1, H, W]), with optional transform applied."""
        x = self.data[idx]
        if self.transform is not None:
            # Transforms should operate on tensors, not PIL/ndarrays, and preserve shape [1, H, W].
            x = self.transform(x)
        # Each element of self.data is a torch.Tensor of shape [1,H,W]
        return x

def get_dataloader(data_dir, batch_size=8, shuffle=True, norm_image=True, num_samples=None):
    """Build a DataLoader for HipMRI 2D slices found in a directory.

    Args:
        data_dir (str): Directory containing NIfTI slice/volume files.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle samples (use True for training).
        norm_image (bool): Apply z-score normalization per slice during load.
        num_samples (int|None): If set, limit to the first `num_samples` files.

    Returns:
        torch.utils.data.DataLoader: Loader over `HipMRIDataset`.
    """
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii')]
    if num_samples:
        file_list = file_list[:num_samples]

    dataset = HipMRIDataset(file_list, norm_image=norm_image, trasnform=trasnform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader



