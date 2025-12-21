import os
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from models.vae import PianoRollResNetVAE

CFG = {
    "sr": 24000,
    "fps": 100,
    "chunk_sec": 10.24,
    "stride_sec": 5.0,
}

class PianoRollDataset(Dataset):
    def __init__(self, dataset_root="preprocessing/roll_tensor"):
        self.piano_roll_paths = []
        dataset_path = Path(dataset_root)
        if not dataset_path.exists():
            raise ValueError(f"Path {dataset_root} does not exist")

        for id_folder in sorted(dataset_path.iterdir()):
            if not id_folder.is_dir(): continue
            pt_files = sorted(id_folder.glob("*.pt"))
            self.piano_roll_paths.extend(pt_files)

    def __len__(self):
        return len(self.piano_roll_paths)

    def __getitem__(self, idx):
        piano_tensor = torch.load(self.piano_roll_paths[idx])
        return piano_tensor

    def get_file_path(self, idx):
        return str(self.piano_roll_paths[idx])


class ChromaLoss(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.chroma_filter = torch.zeros((12, 128), device=device)
        for i in range(128):
            self.chroma_filter[i % 12, i] = 1.0

    def forward(self, pred_logits, target_roll):
        pred_prob = torch.sigmoid(pred_logits)

        pred_flat = pred_prob.squeeze(1).permute(0, 2, 1) # (B, T, 128)
        target_flat = target_roll.squeeze(1).permute(0, 2, 1)

        filter_t = self.chroma_filter.t() # (128, 12)

        pred_chroma = torch.matmul(pred_flat, filter_t)
        target_chroma = torch.matmul(target_flat, filter_t)

        pred_chroma = pred_chroma / (pred_chroma.sum(dim=-1, keepdim=True) + 1e-6)
        target_chroma = target_chroma / (target_chroma.sum(dim=-1, keepdim=True) + 1e-6)

        # Cosine similarity Loss on Chroma
        # here one can switch to an MSE loss
        loss = 1 - F.cosine_similarity(pred_chroma, target_chroma, dim=-1).mean()

        return loss


class WeightedReconLoss(nn.Module):
    def __init__(self, pos_weight, loss_type="l1"):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss_type = loss_type

    def forward(self, pred_logits, target):
        """
        pred_logits: Model output (raw scores, -inf to +inf)
        target: True velocity (0.0 to 1.0)
        """
        pred = torch.sigmoid(pred_logits)

        if self.loss_type == "l1":
            error = torch.abs(pred - target)
        else: # l2
            error = (pred - target) ** 2

        weights = torch.ones_like(target)
        weights[target > 0.05] = self.pos_weight

        return (error * weights).mean()

def calculate_pos_weight(dataloader):
    print("Computing positive class weight...")
    total_pixels = 0
    note_pixels = 0

    for i, batch in enumerate(dataloader):
        if i >= 50: break
        note_pixels += (batch > 0.05).sum().item()
        total_pixels += batch.numel()

    if note_pixels == 0: return 10.0

    # Simple ratio
    ratio = (total_pixels - note_pixels) / note_pixels

    # CLAMP the weights
    safe_weight = min(ratio, 20.0)

    print(f"  Calculated Raw Ratio: {ratio:.2f}")
    print(f"  Safe Clamped Weight: {safe_weight:.2f}")

    return safe_weight

def get_kl_weight(epoch, total_epochs, max_kl=0.0001, warmup_pct=0.2):
    warmup_epochs = int(total_epochs * warmup_pct)
    if epoch < warmup_epochs:
        return max_kl * (epoch / warmup_epochs)
    return max_kl



def train_vae(
    model,
    train_loader,
    val_loader=None,
    epochs=40,
    lr=1e-4,
    device="cuda",
    kl_weight=0.0001,
    checkpoint_dir="ckpts",
    log_dir="logs",
    save_freq=5,
    resume_from=None
):
    # Setup
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=f"{log_dir}/vae_resume{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    if resume_from:
        print(f"Loading checkpoint from: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
        else:
            model.load_state_dict(checkpoint)
            try:
                import re
                match = re.search(r"ep(\d+)", str(resume_from))
                if match:
                    start_epoch = int(match.group(1))
            except:
                print("Could not infer epoch from filename. Starting from epoch 0 (with loaded weights).")

        print(f"Resuming training from Epoch {start_epoch+1}")

        for _ in range(start_epoch):
            scheduler.step()

    raw_pos_weight = calculate_pos_weight(train_loader)

    # Losses
    criterion = WeightedReconLoss(pos_weight=raw_pos_weight, loss_type="l1").to(device)
    chroma_criterion = ChromaLoss(device=device)

    print(f"Starting Training on {device}...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0
        running_chroma = 0.0

        current_kl = get_kl_weight(epoch, epochs, max_kl=kl_weight)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch.to(device)

            recon_logits, mu, log_var = model(x)

            recon_loss = criterion(recon_logits, x)
            chroma_loss = chroma_criterion(recon_logits, x)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

            # Combine loss
            loss = recon_loss + (current_kl * kld_loss) + (0.2 * chroma_loss)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            # Logs for tensorboard
            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld_loss.item()
            running_chroma += chroma_loss.item()

            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'Rec': f"{recon_loss.item():.4f}",
                'KL': f"{kld_loss.item():.2f}",
                'Ch': f"{chroma_loss.item():.2f}"
            })

        # Epoch Stats
        avg_loss = running_loss / len(train_loader)
        avg_recon = running_recon / len(train_loader)
        avg_chrom = running_chroma / len(train_loader)

        writer.add_scalar('epoch/train_loss', avg_loss, epoch)
        writer.add_scalar('epoch/recon_loss', avg_recon, epoch)
        writer.add_scalar('epoch/chroma_loss', avg_chrom, epoch)

        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_l1_err = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    recon_logits, mu, log_var = model(x)

                    # Val Loss
                    r_loss = criterion(recon_logits, x)
                    k_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
                    c_loss = chroma_criterion(recon_logits, x)
                    v_loss = r_loss + (current_kl * k_loss) + (0.2 * c_loss)

                    val_loss += v_loss.item()

                    # Pure L1 error on notes only !
                    pred = torch.sigmoid(recon_logits)
                    mask = (x > 0.05)
                    if mask.sum() > 0:
                        val_l1_err += torch.abs(pred[mask] - x[mask]).mean().item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_err = val_l1_err / len(val_loader)

            print(f"Val Loss: {avg_val_loss:.4f} | Avg Note Velocity Error: {avg_val_err:.4f}")
            writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")

        scheduler.step()

        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f"{checkpoint_dir}/vae_ep{epoch+1}.pth")

    writer.close()
    return model

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 2 # change to more if the user has more gpu memory available
    # as we only have 6 gb, we are limited.
    torch.manual_seed(1234)

    data_dir = "/mnt/ssd-samsung/atiam/projet_ml/roll_tensor/"
    full_dataset = PianoRollDataset(dataset_root=data_dir)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PianoRollResNetVAE(
        input_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        latent_dim=4
    )

    resume_path = "ckpts/vae_ep20.pth"
    train_vae(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=40,
        lr=1e-3,
        device=DEVICE,
        resume_from=resume_path
    )
