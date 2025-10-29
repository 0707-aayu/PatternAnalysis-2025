# sanity_checks.py
import os, re, argparse, random, hashlib, json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from dataset import get_dataloader
from modules import VQVAE
from utils import calc_ssim, read_yaml_file

# --------------------------
# Helpers for leakage checks
# --------------------------
def list_nii(root):
    return sorted([os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.nii', '.nii.gz'))])

def sha1_of_array(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(arr.tobytes())
    return h.hexdigest()

def load_volume_signature(path):
    """
    Returns a signature dict for a .nii/.nii.gz file capturing:
      - voxel hash (sha1)
      - shape
      - affine (rounded)
    Also returns a 'patient_id' guessed from filename or parent folder.
    """
    img = nib.load(path)
    data = img.get_fdata()
    # robust rounding to avoid tiny float diffs
    aff = np.round(img.affine, 4)
    sig = {
        "sha1": sha1_of_array(data.astype(np.float32)),
        "shape": tuple(data.shape),
        "affine": sha1_of_array(aff.astype(np.float32)),
    }
    # heuristics to infer patient/volume id
    fname = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    # try a few patterns commonly seen
    m = (
        re.search(r'(patient|pt|subj|subject|case)[-_]?(\d+)', fname, re.I)
        or re.search(r'(patient|pt|subj|subject|case)[-_]?(\d+)', parent, re.I)
        or re.search(r'([A-Za-z]*\d{3,})', fname)
    )
    if m:
        patient_id = m.group(0).lower()
    else:
        # fallback: stem without slice index if present
        patient_id = re.sub(r'_slice\d+|\-slice\d+|\d{1,4}\.nii(\.gz)?$', '', fname.lower())
    return sig, patient_id

def check_disjoint_sets(train_files, val_files, test_files):
    sets = {
        "train": set(map(os.path.basename, train_files)),
        "val":   set(map(os.path.basename, val_files)),
        "test":  set(map(os.path.basename, test_files)),
    }
    collisions = {
        "train∩val": list(sets["train"].intersection(sets["val"])),
        "train∩test": list(sets["train"].intersection(sets["test"])),
        "val∩test": list(sets["val"].intersection(sets["test"])),
    }
    return collisions

def detect_duplicates(files):
    """
    Detect perfect duplicates within a split (same voxel sha1).
    """
    seen = {}
    dups = []
    for p in files:
        try:
            sig, _ = load_volume_signature(p)
            key = (sig["sha1"], sig["shape"], sig["affine"])
            if key in seen:
                dups.append((seen[key], p))
            else:
                seen[key] = p
        except Exception as e:
            print(f"[WARN] Could not hash {p}: {e}")
    return dups

def patient_overlap(train_files, val_files, test_files):
    """
    Patient/volume ID overlap across splits (slice leakage protection).
    """
    def ids(files):
        out = set()
        for p in files:
            try:
                _, pid = load_volume_signature(p)
                out.add(pid)
            except Exception:
                pass
        return out

    tr_ids, va_ids, te_ids = ids(train_files), ids(val_files), ids(test_files)
    return {
        "train∩val": sorted(list(tr_ids & va_ids)),
        "train∩test": sorted(list(tr_ids & te_ids)),
        "val∩test": sorted(list(va_ids & te_ids)),
    }

# --------------------------------------------
# Overfitting quick check (mini-batch SSIM gap)
# --------------------------------------------
@torch.no_grad()
def quick_overfit_check(config, device, n_samples=64, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    model = VQVAE(**config["model_parameters"]).to(device)
    weights = os.path.join(config.get("log_dir", "logs"), "best_model.pth")
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    criterion = nn.MSELoss()

    # Use your loaders (no transforms needed for SSIM)
    train_loader = get_dataloader(config["train_dataset_dir"], batch_size=8, shuffle=True)
    val_loader   = get_dataloader(config["val_dataset_dir"],   batch_size=8, shuffle=False)

    def batch_stats(loader, limit):
        total_ssim, total_loss, count = 0.0, 0.0, 0
        for batch in loader:
            batch = batch.to(device).float()
            recon, commit, _ = model(batch)
            loss = criterion(recon, batch) + commit
            # per-image SSIM
            for i in range(batch.size(0)):
                ssim_val = calc_ssim(batch[i,0].cpu().numpy(), recon[i,0].cpu().numpy())
                total_ssim += ssim_val
                total_loss += loss.item()
                count += 1
                if count >= limit:
                    break
            if count >= limit:
                break
        return total_ssim / count, total_loss / count

    train_ssim, train_loss = batch_stats(train_loader, n_samples)
    val_ssim,   val_loss   = batch_stats(val_loader,   n_samples)

    gap = train_ssim - val_ssim
    print("\n=== Overfitting Quick Check (mini-batch) ===")
    print(f"Train SSIM (≈{n_samples} imgs): {train_ssim:.4f}")
    print(f"Val   SSIM (≈{n_samples} imgs): {val_ssim:.4f}")
    print(f"SSIM Gap (Train - Val):        {gap:+.4f}")
    print(f"Train Loss (avg): {train_loss:.4f} | Val Loss (avg): {val_loss:.4f}")

    # Simple rule-of-thumb interpretation
    if gap > 0.02:
        print("Interpretation: Noticeable gap (>0.02) → potential overfitting. Consider stronger aug/regularization or earlier stop.")
    elif gap < -0.01:
        print("Interpretation: Val > Train (negative gap) → training regularization or noisy train; not typical overfitting.")
    else:
        print("Interpretation: Small gap → no strong evidence of overfitting.")

# -----------------
# Main entry point
# -----------------
def main():
    ap = argparse.ArgumentParser(description="Leakage & Overfitting Sanity Checks")
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir",   required=True)
    ap.add_argument("--test_dir",  required=True)
    ap.add_argument("--config",    required=True, help="config.yaml with model params and log_dir")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample_overfit_n", type=int, default=64)
    args = ap.parse_args()

    # 1) File-level disjointness
    tr_files = list_nii(args.train_dir)
    va_files = list_nii(args.val_dir)
    te_files = list_nii(args.test_dir)

    print(f"\nCounts → train:{len(tr_files)}  val:{len(va_files)}  test:{len(te_files)}")

    print("\n[1] Checking filename collisions across splits…")
    collisions = check_disjoint_sets(tr_files, va_files, te_files)
    print(json.dumps(collisions, indent=2))
    if any(len(v) for v in collisions.values()):
        print(" Found filename overlap across splits → leakage risk.")

    print("\n[2] Checking patient/volume ID overlap across splits…")
    pid_overlap = patient_overlap(tr_files, va_files, te_files)
    print(json.dumps(pid_overlap, indent=2))
    if any(len(v) for v in pid_overlap.values()):
        print("Same patient/volume appears in multiple splits → slice leakage risk.")

    print("\n[3] Checking exact duplicate volumes within each split…")
    for name, files in [("train", tr_files), ("val", va_files), ("test", te_files)]:
        dups = detect_duplicates(files)
        if dups:
            print(f" {name}: found {len(dups)} duplicate pairs (same voxel hash). Examples:")
            for a,b in dups[:5]:
                print("   ", a, "<->", b)
        else:
            print(f" {name}: no exact duplicates detected.")

    # 2) Overfitting quick check (uses your trained model)
    print("\n[4] Overfitting mini-batch diagnostic…")
    config = read_yaml_file(args.config)
    quick_overfit_check(config, args.device, n_samples=args.sample_overfit_n)

if __name__ == "__main__":
    main()
