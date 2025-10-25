import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

def pad_to_fixed_size(tensor, target_height, target_width):
    """
    Pads a tensor to the target height and width with zeros on right and bottom sides.
    Tensor shape: [1, H, W]
    """
    _, h, w = tensor.shape
    pad_h = target_height - h
    pad_w = target_width - w
    return F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)

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
    
    # First pass: load and normalize; track max H/W
    for filepath in file_list:
        nii_img = nib.load(filepath)
        img = nii_img.get_fdata()

        if img.ndim == 2:
            slice_2d = img
            if norm_image:
                slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)
            t = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
            temp_data.append(t)

        elif img.ndim == 3:
            for i in range(img.shape[2]):
                slice_2d = img[:, :, i]
                if norm_image:
                    slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)
                t = torch.from_numpy(slice_2d.astype(np.float32)).unsqueeze(0)
                temp_data.append(t)

        else:
            print(f"Skipping {filepath}: unexpected shape {img.shape}")

    # Track maximum dimensions
    for t in temp_data:
        _, h, w = t.shape
        max_height = max(max_height, h)
        max_width  = max(max_width,  w)

    # Second pass: pad to uniform size
    data = [pad_to_fixed_size(t, max_height, max_width) for t in temp_data]

    print(f"Total usable 2D slices loaded: {len(data)} with uniform size ({max_height}, {max_width})")
    return data


def calc_ssim(x, y):
    """
    Calculate batch mean SSIM between two tensors x and y.

    Args:
        x (torch.Tensor): Batch of images [B, 1, H, W]
        y (torch.Tensor): Batch of images [B, 1, H, W]

    Returns:
        float: Mean SSIM over the batch.
    """
    x = x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    batch_size = x.shape[0]
    ssim_values = []
    for i in range(batch_size):
        ssim_val = ssim(x[i, 0], y[i, 0], data_range=x[i, 0].max() - x[i, 0].min())
        ssim_values.append(ssim_val)

    mean_ssim = np.mean(ssim_values)
    return mean_ssim
