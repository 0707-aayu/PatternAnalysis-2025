import os
from torch.utils.data import Dataset, DataLoader
import torch
from utils import load_data_2d

class HipMRIDataset(Dataset):
    """
    PyTorch Dataset for HipMRI processed 2D slices
    """

    def __init__(self, file_list, norm_image=True, transform=None):
        """
        Args:
            file_list (list): List of full paths to .nii files
            norm_image (bool): Whether to normalize each slice
        """
        self.data = load_data_2d(file_list, norm_image=norm_image)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        # Each element of self.data is a torch.Tensor of shape [1,H,W]
        return x

def get_dataloader(data_dir, batch_size=8, shuffle=True, norm_image=True, num_samples=None):
    """
    Helper function to get DataLoader for HipMRI 2D slices.

    Args:
        data_dir (str): Directory containing the .nii slice files
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        norm_image (bool): Normalize image slices
        num_samples (int or None): Optional number of samples to limit loading

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii')]
    if num_samples:
        file_list = file_list[:num_samples]

    dataset = HipMRIDataset(file_list, norm_image=norm_image, trasnform=trasnform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


