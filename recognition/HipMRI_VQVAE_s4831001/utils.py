import nibabel as nib
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def load_data_2d(file_list, norm_image=True):
    """
    Loads NIfTI files and returns a list of normalized 2D torch tensors.

    Args:
        file_list (list): list of file paths to .nii or .nii.gz files
        norm_image (bool): whether to normalize image intensities to zero mean and unit std

    Returns:
        data (list): list of 2D image tensors (torch.Tensor) with shape [1, H, W]
    """
    data = []
    for filepath in file_list:
        nii_img = nib.load(filepath)
        img = nii_img.get_fdata()

        # Convert 3D volume to 2D slices (slice along last dimension)
        for i in range(img.shape[2]):
            slice_2d = img[:, :, i]

            if norm_image:
                slice_2d = (slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-8)

            slice_tensor = torch.from_numpy(slice_2d.astype(np.float32))
            data.append(slice_tensor.unsqueeze(0))  # Add channel dim [1, H, W]

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
