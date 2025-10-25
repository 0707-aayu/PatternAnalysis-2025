import os
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloader
from modules import VQVAE
from utils import calc_ssim, read_yaml_file

def predict(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = config['model_parameters']
    pretrained_path = config['pretrained_path']
    test_dataset_dir = config['test_dataset_dir']
    save_dir = config.get('save_dir', 'predictions')
    os.makedirs(save_dir, exist_ok=True)

    test_loader = get_dataloader(test_dataset_dir, batch_size=1, shuffle=False)

    # Load trained model
    model = VQVAE(**model_params).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model.eval()
    criterion = nn.MSELoss()

    total_ssim = 0.0
    total_loss = 0.0
    num_images = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Predicting"), 1):
            batch = batch.to(device).float()
            reconstructed, commitment_loss = model(batch)
            recon_loss = criterion(reconstructed, batch)
            loss = recon_loss + commitment_loss

            original_image = batch[0, 0].cpu().numpy()
            reconstructed_image = reconstructed[0, 0].cpu().numpy()
            ssim_value = calc_ssim(original_image, reconstructed_image)

            total_ssim += ssim_value
            total_loss += loss.item()
            num_images += 1

    print(f"\nAverage test SSIM: {total_ssim/num_images:.4f}")
    print(f"Average test loss: {total_loss/num_images:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict using VQVAE model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()
    config = read_yaml_file(args.config)
    predict(config)
