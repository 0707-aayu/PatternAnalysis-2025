import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision import transforms
import yaml

def pad_to_fixed_size(tensor, target_height, target_width):
    """
    Pads a tensor to the target height and width with zeros on right and bottom sides.
    Tensor shape: [1, H, W]
    """
    _, h, w = tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

class GaussianNoiseTransform:
    def __init__(self, std=0.05, p=0.5):
        self.std = std
        self.p = p
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return x + torch.randn_like(x) * self.std
        return x

def load_data_2d(file_list, norm_image=True):
    """
    Loads NIfTI files and returns a list of normalized 2D torch tensors.

    Args:
        file_list (list): list of file paths to .nii or .nii.gz files
        norm_image (bool): whether to normalize image intensities to zero mean and unit std

    Returns:
        data (list): list of 2D image tensors (torch.Tensor) with shape [1, H, W]
    """
    temp_data = []
    max_height, max_width = 0, 0
    
    # First pass: load data, normalize, find max height/width
    for filepath in file_list:
        try:
            nii_img = nib.load(filepath)
            img = nii_img.get_fdata()
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

    # Determine max height and width
    for tensor in temp_data:
        _, h, w = tensor.shape
        if h > max_height:
            max_height = h
        if w > max_width:
            max_width = w

    # Second pass: pad all tensors to max height and width
    data = [pad_to_fixed_size(tensor, max_height, max_width) for tensor in temp_data]

    print(f"Total usable 2D slices loaded: {len(data)} with uniform size ({max_height}, {max_width})")
    return data



def calc_ssim(x, y):
    """
    Calculate batch mean SSIM between two numpy arrays x and y.

    Args:
        x (np.ndarray): Batch image [H, W] or [1, H, W]
        y (np.ndarray): Batch image [H, W] or [1, H, W]

    Returns:
        float: SSIM score.
    """
    ssim_val = ssim(x.squeeze(), y.squeeze(), data_range=x.max() - x.min())
    return ssim_val



def get_transforms(transform_list):
    """
    Build an augmentation pipeline that works on tensors shaped [1, H, W].
    (We don't add ToTensor/Normalize because you already return tensors and
    z-score each slice in load_data_2d.)
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
