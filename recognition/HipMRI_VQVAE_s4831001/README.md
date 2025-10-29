# Vector-Quantized Variational Autoencoder (VQ-VAE) for HipMRI Prostate Cancer Radiotherapy Reconstruction

## Description

This repository implements a **Vector-Quantized Variational Autoencoder (VQ-VAE)** for reconstructing 2D pelvic MRI slices from the CSIRO HipMRI dataset.  
The model learns a discrete latent representation using a codebook of embeddings, allowing high-fidelity and interpretable reconstructions suitable for medical research applications.  
Reconstruction performance is evaluated using the Structural Similarity Index (SSIM) — with values above 0.6 considered clinically acceptable.

**Reference:** [van den Oord et al., 2017 – Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

---

## VQ-VAE Description

A **Vector-Quantized Variational Autoencoder (VQ-VAE)** is a generative model that extends the traditional VAE by introducing vector quantization, which replaces the continuous latent space with a discrete codebook of learned embeddings.  
Each latent vector is mapped to its nearest codebook entry, producing compact and interpretable latent representations that improve reconstruction quality and stability — particularly valuable in medical imaging, where structural fidelity and anatomical consistency are essential.

---

## Implementation Overview

### Architecture

![Architecture](images/Architecture.png)

The VQ-VAE comprises three components:

- **Encoder** – Compresses input MRI slices into latent representations using convolutional layers with BatchNorm, ReLU, and residual stacks to maintain gradient flow.  
- **Vector Quantiser (VQ)** – Replaces each latent vector with its nearest codebook embedding, discretising the latent space via nearest-neighbour lookup.  
- **Decoder** – Reconstructs the original image from quantised embeddings using transposed convolutions and residual blocks.

Residual connections throughout the network ensure stable training and preserve fine structural details.

---

## Loss Function & Optimisation

The model jointly optimises three loss terms:

![Loss Function Table ](images/Loss_function_table.png)

## Key Insight

By discretising the latent space, the VQ-VAE avoids the “posterior collapse” issue common in VAEs and provides a stable, interpretable representation of MRI slices — capturing both global anatomical structures and local variations crucial for radiotherapy analysis.

---

## Environment & Setup

This repository includes a `requirements.txt` file to ensure reproducibility of the Python environment and dependencies.

### Option A: Using `venv`

python -m venv .venv

Activate (Windows)
.venv\Scripts\activate

Activate (Unix/MacOS)
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt


### Option B: Using `conda`

conda create -n vqvae_env python=3.10 -y

conda activate vqvae_env

pip install -r requirements.txt

---

## Data Integrity & Overfitting Verification

To confirm reliability, a diagnostic script (`sanity.py`) was used.

### Data Leakage Verification

- No filename or patient ID overlaps across splits.
- No duplicate slices or volumes within any split.

### Overfitting Diagnostic

| Metric   | Train    | Validation | Gap      | Interpretation                |
|----------|----------|------------|----------|-------------------------------|
| SSIM     | 0.9365   | 0.9168     | +0.0197  | Minimal gap — no strong overfitting |
| MSE Loss | 0.0241   | 0.0375     | +0.0134  | Stable generalization          |

Validation and test SSIM (~0.935 ± 0.011) align closely, confirming robust generalization and balanced codebook utilisation (perplexity ≈ 270).

---

**Distribution Plot:**  
![SSIm Loss Distribution ](images/ssim_loss_distribution.png)  
Shows tightly clustered SSIM ≈ 0.935 (σ ≈ 0.011) and MSE ≈ 0.024, demonstrating uniform reconstruction quality.

**To rerun:**
python sanity.py --train_dir keras_slices_data/keras_slices_train --val_dir keras_slices_data/keras_slices_validate --test_dir keras_slices_data/keras_slices_test --config config.yaml

---

## Configuration Overview

This repository uses a YAML-based configuration system (`config.yaml`) for full reproducibility across both training and inference.

All key hyperparameters, dataset paths, and augmentation settings are declared in this single file and dynamically loaded by both `train.py` and `predict.py`.  
By modifying `config.yaml`, you can:
- Re-train the model with different architectures or learning settings.
- Adjust augmentation intensity and dataset directories.
- Run predictions from pre-trained checkpoints without editing the codebase.

### Configuration Parameters

All parameters are sourced from the configuration dictionary parsed from `config.yaml`. Model parameters are grouped under `model_parameters`, while the rest control training, data loading, and prediction.

#### Model Architecture Parameters (VQ-VAE)

| Parameter        | Description                                                  |
|------------------|--------------------------------------------------------------|
| `in_channels`    | Number of input channels (e.g., 1 for grayscale MRI slices). |
| `hidden_channels`| Depth of convolutional feature maps in encoder/decoder.     |
| `res_channels`   | Feature width within residual blocks.                        |
| `nb_res_layers`  | Number of residual blocks stacked in encoder/decoder.       |
| `embed_dim`      | Dimension of each embedding vector in the codebook.         |
| `nb_entries`     | Codebook size — total number of discrete embeddings.        |
| `downscale_factor`| Factor by which encoder reduces spatial dimensions (power of 2).|

#### Training Parameters (`train.py`)

| Parameter      | Description                                                      |
|----------------|------------------------------------------------------------------|
| `batch_size`   | Number of samples per training iteration (used in both train/val). |
| `num_epochs`   | Total number of passes over the dataset.                         |
| `learning_rate`| Initial learning rate for the Adam optimiser.                    |
| `weight_decay` | L2 regularisation coefficient (default = 0).                    |
| `log_dir`      | Directory for logs, checkpoints, and visualisations.            |

Logging Behaviour (hard-coded):
- Example reconstructions saved every 5 epochs → `logs/images/epoch_X.png`.
- Training metrics auto-saved → `metrics.json` and `training_curves.png`.
- Best model saved as `logs/best_model.pth` (lowest validation loss).

#### Dataset & Data Loading Parameters

| Parameter          | Description                                        |
|--------------------|----------------------------------------------------|
| `train_dataset_dir`| Directory with training `.nii` slices.            |
| `val_dataset_dir`  | Directory with validation `.nii` slices.          |
| `test_dataset_dir` | Directory with test `.nii` slices (evaluation).   |
| `train_num_samples`/`val_num_samples`/`test_num_samples` | Optionally limit loaded samples per split.|
| `train_transforms` | Data augmentations: `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, `RandomAffine`, `GaussianNoise`. |
| `val_test_transforms` | Augmentations for validation/test data (usually empty). |
| `norm_image`       | Flag to apply z-score normalisation per slice.     |

#### Prediction Parameters (`predict.py`)

| Parameter         | Description                                   |
|-------------------|-----------------------------------------------|
| `pretrained_path` | Path to model checkpoint (e.g., `logs/best_model.pth`). |
| `save_dir`        | Output directory for reconstructed images and plots. |
| `test_dataset_dir`| Directory containing test slices for inference. |
| `val_test_transforms` | Optional augmentations for inference (usually empty). |

---

## Code Structure

The project is designed with modularity to enhance clarity, maintainability, and reproducibility across various components:

- `modules.py`: Implements the full VQ-VAE architecture, including ResidualBlock, ResidualStack for stable deep feature extraction, VectorQuantizer for discrete codebook lookups and loss computation, and the unified VQVAE class that connects encoder, quantizer, and decoder into an end-to-end model.

- `dataset.py`: Manages data loading and preprocessing for 2D MRI slices stored in NIfTI files. Defines HipMRIDataset to normalize slices, and get_dataloader() to build PyTorch loaders for training, validation, and testing, with support for data augmentation driven by config.yaml.

- `train.py`: Contains the training loop, initializing the model, optimizer, and data loaders, performing forward and backward passes, and calculating reconstruction, commitment, and quantization losses over multiple epochs. Tracks SSIM and perplexity, saves best checkpoints (logs/best_model.pth), and generates training curves.

- `predict.py`: Handles inference by loading a pre-trained checkpoint, reconstructing MRI slices, computing SSIM and MSE, and saving reconstructed images, comparison grids, and distribution plots (ssim_loss_distribution.png) to a predictions/ directory.

- `utils.py`: Provides shared utilities for YAML parsing, NIfTI loading, SSIM calculation, visualization, and augmentation construction, including custom transforms such as Gaussian noise.

- `config.yaml`: Central configuration file that specifies all parameters—model architecture, hyperparameters, dataset paths, and augmentations—permitting full experiment reproducibility without modifying code.

---

## Hyperparameter Tuning Summary

### Learning Rate Study

| LR     | Train Loss | Val Loss | Train SSIM | Val SSIM | Observation        |
|--------|------------|----------|------------|----------|--------------------|
| 0.0001 | 0.0817     | 0.0753   | 0.642      | 0.646    | Too slow           |
| 0.0005 | 0.0073     | 0.0090   | 0.890      | 0.896    | Minor instability  |
| **0.001**  | **0.0061**     | **0.0053**   | **0.906**      | **0.917**    | **Best balance**       |
| 0.005  | 0.0052     | 0.0048   | 0.907      | 0.912    | Oscillatory        |

### Hyperparameter Sensitivity (Validation Set)

| Config | Hidden Ch. | Embed Dim | Codebook Size | Val SSIM | Val Loss | Perplexity      | Observation     |
|--------|------------|-----------|---------------|----------|----------|-----------------|-----------------|
| A      | 64         | 32        | 256           | 0.865    | 0.038    | 220 ± 25        | Underfitting    |
| **B**      | **128**        | **64**        | **512**           | **0.917**    | **0.029**    | **275 ± 18**        | **Best balance**    |
| C      | 256        | 128       | 512           | 0.923    | 0.028    | 320 ± 22        | Heavier model   |
| D      | 128        | 64        | 1024          | 0.910    | 0.031    | 190 ± 35        | Partial collapse|

### Augmentation Effect

| Setting           | Val SSIM | Observation        |
|-------------------|----------|--------------------|
| Without Augmentation | 0.937    | Slight overfitting  |
| **With Augmentation**  | **0.959**    | **Better generalization**|

### Downscale Factor

| Factor | Val SSIM | Observation         |
|--------|----------|---------------------|
| **4**      | **0.917**    | **Optimal balance**      |
| 8      | 0.832    | Detail loss         |
| 16     | 0.595    | Severe degradation   |

### Optimizer Comparison

| Optimizer | LR    | SSIM | Observation     |
|-----------|-------|------|-----------------|
| **Adam**      | **0.001** | **0.917**| **Stable, smooth**  |
| SGD       | 0.001 | 0.523| Noisy codebook  |

---

## Results & Discussion

• Training and validation curves converged smoothly with no signs of overfitting (SSIM gap < 0.02).  
• The final model achieved:  
  - **Mean SSIM:** 0.935 ± 0.011  
  - **Mean Reconstruction Loss:** 0.024  
  - **Codebook Perplexity:** ≈ 270–300  
• Reconstructions retained sharp tissue boundaries and anatomical structure consistency.  
• Data augmentation notably improved generalization and robustness to scanner variations.  

---

### Training Progress

The model demonstrated stable convergence across loss, SSIM, and codebook utilization:

![Training Curves](images/final_training_curves.png)

---

### Reconstruction Quality Across Epochs

Reconstruction fidelity improved gradually as training progressed:

| Epoch 5 | Epoch 10 | Epoch 15 |
|----------|-----------|-----------|
| ![Epoch 5](images/epoch_5.png) | ![Epoch 10](images/epoch_10.png) | ![Epoch 15](images/epoch_15.png) |

| Epoch 20 | Epoch 25 | Epoch 30 |
|-----------|-----------|-----------|
| ![Epoch 20](images/epoch_20.png) | ![Epoch 25](images/epoch_25.png) | ![Epoch 30](images/epoch_30.png) |

---

## Sanity Check & Console Outputs

After model training and testing, additional console-based evaluations were conducted to verify learning stability and ensure no data leakage or overfitting.

### Training Phase Logs

The VQ-VAE training progressed steadily across 30 epochs, achieving balanced convergence for loss, SSIM, and codebook perplexity.

![Training Start](images/training_start.png)
![Training Output](images/train_output.png)

---

### Test Phase Results

Testing confirmed the model’s high structural similarity and low reconstruction loss across 540 test slices.

![Test Output](images/test_output.png)
![Test Output (Detailed)](images/test_output_1.png)

---

### Sanity Check: Overfitting & Data Leakage Validation

A lightweight diagnostic script (`sanity.py`) was executed post-training to confirm generalization performance.  
It compared SSIM and loss over small batches from the train and validation splits.

**Observation:**  
The SSIM gap between training and validation was approximately **+0.0197**, indicating *no strong evidence of overfitting*.

![Sanity Check Output](images/sanity_check_output.png)

---


## Model Behavior Insights

The reconstructed MRI slices display slight smoothness due to the Mean Squared Error (MSE) objective, which penalizes pixel-level deviations and causes the decoder to predict averaged intensities.  
Combined with vector quantization and transposed convolutions, this smooths high-frequency edges but maintains globally consistent anatomy — a desirable property in medical imaging where structural fidelity outweighs pixel sharpness.

The slightly higher validation SSIM compared to training does not indicate overfitting. Instead, it results from stochastic augmentations (flips, affine shifts, Gaussian noise) applied only during training, which make reconstruction more challenging.  
Validation data, being unaugmented, yield marginally better SSIM scores. This small gap (~0.02) demonstrates strong generalization and stable learning.

---

## Future Work

- Hierarchical / Multi-Level VQ-VAE for richer latent hierarchies
- 3D or Temporal Extensions to exploit volumetric continuity
- Adaptive Codebook and Entropy Regularization
- Hybrid Perceptual Losses to reduce smoothness
- Clinical Evaluation & Cross-Dataset Validation for robustness

---

## Conclusion

This implementation of VQ-VAE for HipMRI reconstruction achieved strong, reproducible results —  
**SSIM ≈ 0.935 ± 0.011, Loss ≈ 0.024, Perplexity ≈ 280** — with clean generalization and no overfitting.  
Through careful hyperparameter tuning and augmentation, the model preserved structural accuracy while maintaining computational efficiency.  
It establishes a robust foundation for discrete representation learning in medical imaging, paving the way for perceptually enhanced and clinically interpretable reconstruction frameworks.


