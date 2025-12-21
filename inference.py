import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Models
from models.transformer import Pop2PianoDiT_CrossAttn
from models.conditioner import AudioConditioner
from models.vae import PianoRollResNetVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/mnt/ssd-samsung/atiam/projet_ml/roll_tensor"
DIT_CKPT = "/mnt/ssd-samsung/atiam/projet_ml/ckpts/dit_ep150.pth"
VAE_CKPT = "/mnt/ssd-samsung/atiam/projet_ml/ckpts/best_model_vae.pth"

LATENT_SCALE = 0.18215
SR = 24000
CHUNK_SEC = 10.24
FPS = 100

TIMESTEPS = 1000
DDIM_STEPS = 50

class DiffusionSampler:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2):
        self.model = model
        self.timesteps = timesteps

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def q_sample(self, x0, t, noise):
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise

    @torch.no_grad()
    def ddim_sample(self, audio_context, shape):
        x = torch.randn(shape, device=DEVICE)

        step_size = self.timesteps // DDIM_STEPS
        steps = list(range(0, self.timesteps, step_size))[::-1]

        for i, t in enumerate(steps):
            t_tensor = torch.full((shape[0],), t, device=DEVICE, dtype=torch.long)

            eps = self.model(x, t_tensor, audio_context)

            alpha_t = self.alpha_bar[t]
            alpha_prev = self.alpha_bar[steps[i + 1]] if i < len(steps) - 1 else torch.tensor(1.0, device=DEVICE)

            x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            x = torch.sqrt(alpha_prev) * x0 + torch.sqrt(1 - alpha_prev) * eps

        return x / LATENT_SCALE

def load_pairs(data_root):
    data_root = Path(data_root)
    pairs = []

    for latent_path in data_root.rglob("*_latent.pt"):
        wav_path = latent_path.with_name(
            latent_path.name.replace("_latent.pt", ".wav")
        )

        if not wav_path.exists():
            continue

        pairs.append((latent_path, wav_path))

    if len(pairs) == 0:
        raise RuntimeError("No valid (latent, wav) pairs found!")

    return pairs



def load_audio_chunk(wav_path):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)

    samples = int(CHUNK_SEC * SR)
    wav = wav[:, :samples]
    if wav.shape[1] < samples:
        wav = F.pad(wav, (0, samples - wav.shape[1]))

    return wav.squeeze(0).numpy()

def run_inference_eval(num_samples=100):
    print("Loading models...")

    dit = Pop2PianoDiT_CrossAttn(
        input_size=(16, 128),
        patch_size=4,
        in_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        audio_dim=768
    ).to(DEVICE).eval()
    dit_ckpt = torch.load(DIT_CKPT, map_location=DEVICE)
    dit.load_state_dict(dit_ckpt['model'])

    conditioner = AudioConditioner().to(DEVICE).eval()

    vae = PianoRollResNetVAE(
        input_channels=1,
        base_channels=32,
        channel_mults=(1, 2, 4, 8),
        latent_dim=4
    ).to(DEVICE).eval()
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=DEVICE))

    sampler = DiffusionSampler(dit)

    pairs = load_pairs(DATA_ROOT)[:num_samples]

    eps_mse = []
    x0_mse = []
    decoded_mse = []

    for latent_path, audio_path in tqdm(pairs, desc="Evaluating"):
        x0 = torch.load(latent_path).to(DEVICE)
        x0 = x0.unsqueeze(0) * LATENT_SCALE

        audio_np = load_audio_chunk(audio_path)
        inputs = conditioner.process_audio([audio_np], sampling_rate=SR)
        with torch.no_grad():
            audio_ctx = conditioner(inputs["input_values"].to(DEVICE))

        t = torch.randint(0, TIMESTEPS, (1,), device=DEVICE)
        noise = torch.randn_like(x0)
        x_t = sampler.q_sample(x0, t, noise)

        eps_pred = dit(x_t, t, audio_ctx)
        eps_mse.append(F.mse_loss(eps_pred, noise).item())

        # full sampling eval
        x_gen = sampler.ddim_sample(audio_ctx, x0.shape)
        x0_mse.append(F.mse_loss(x_gen, x0).item())

        # decoding x0 with decoder
        with torch.no_grad():
            x0_decoded = vae.decode(x0)
            xpred_decoded = vae.decode(x_gen)

        decoded_mse.append(F.mse_loss(xpred_decoded, x0_decoded).item())

    print("Evaluation Results")
    print(f"epsilon mse : {np.mean(eps_mse):.6f}")
    print(f"x0 mse : {np.mean(x0_mse):.6f}")
    print(f"piano roll mse : {np.mean(decoded_mse):.6f}")


if __name__ == "__main__":
    run_inference_eval(num_samples=20)
