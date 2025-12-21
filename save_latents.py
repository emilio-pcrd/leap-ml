import os
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
sys.path.insert(0, "../")
from models.vae import PianoRollResNetVAE
from train_vae import PianoRollDataset
from tqdm import tqdm


def save_latents(data_dir, output_dir, model, device):
    Path(output_dir).mkdir(exist_ok=True)

    model.eval()

    dataset = PianoRollDataset(data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    pbar = tqdm(loader, desc="Saving latents")

    with torch.no_grad():
        for idx, batch in enumerate(pbar):
            x = batch.to(device)  # (1, 1, 128, 1024)

            mus, sigmas = model.encode(x)
            z_latents = model.reparameterize(mus, sigmas).squeeze(0)  # (C, H, W)

            assert z_latents.ndim == 3, "Latent should be 3D (C, H, W)"

            original_path = dataset.get_file_path(idx)
            latent_path = original_path.replace('.pt', '_latent.pt')

            if output_dir != data_dir:
                rel_path = os.path.relpath(original_path, data_dir)
                latent_path = os.path.join(output_dir, rel_path.replace('.pt', '_latent.pt'))

                Path(latent_path).parent.mkdir(parents=True, exist_ok=True)

            torch.save(z_latents.cpu(), latent_path)

            pbar.set_postfix({'saved': Path(latent_path).name})



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PianoRollResNetVAE(base_channels=32, channel_mults=(1, 2, 4, 8))

    checkpoint_path = "./ckpts/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found, using untrained model")

    model = model.to(device)

    # Save latents
    data_dir = "/mnt/ssd-samsung/atiam/projet_ml/roll_tensor/"
    output_dir = data_dir

    save_latents(data_dir, output_dir, model, device)
