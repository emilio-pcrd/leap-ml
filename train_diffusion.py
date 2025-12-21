import math
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.conditioner import AudioConditioner, note_count_module


class DiffusionDataset(Dataset):
    def __init__(self, dataset_root="preprocessing/roll_tensor"):
        self.piano_roll_paths_latents = []
        self.audio_paths = []

        dataset_path = Path(dataset_root)
        if not dataset_path.exists():
            raise ValueError(f"Le chemin {dataset_root} n'existe pas")

        print("Chargement du dataset...")
        for id_folder in sorted(dataset_path.iterdir()):
            if not id_folder.is_dir():
                continue

            pt_files = sorted(id_folder.glob("*_latent.pt"))
            wav_files = sorted(id_folder.glob("*.wav"))

            if len(pt_files) != len(wav_files):
                print(f"Attention: {id_folder.name} a {len(pt_files)} .pt et {len(wav_files)} .wav")

            min_len = min(len(pt_files), len(wav_files))
            self.piano_roll_paths_latents.extend(pt_files[:min_len])
            self.audio_paths.extend(wav_files[:min_len])

        print(f"Dataset chargé : {len(self.piano_roll_paths_latents)} paires (piano_roll, audio) trouvées")

    def __len__(self):
        return len(self.piano_roll_paths_latents)

    def _get_complexity_class(self, note_count):
        """
        Maps raw note count to complexity classes (0-4).
        """
        if note_count < 30:
            return 0  # Very Simple
        elif note_count < 40:
            return 1  # Simple
        elif note_count < 50:
            return 2  # Medium
        elif note_count < 60:
            return 3  # Complex
        else:
            return 4  # Very Complex / Virtuoso

    def __getitem__(self, idx):
        latent_tensor = torch.load(self.piano_roll_paths_latents[idx])
        audio_tensor, sr_ = torchaudio.load(self.audio_paths[idx])
        piano_tensor = torch.load(str(self.piano_roll_paths_latents[idx]).replace('_latent.pt', '.pt'))

        raw_count = note_count_module(piano_tensor)
        complexity_label = self._get_complexity_class(raw_count)
        class_tensor = torch.tensor(complexity_label, dtype=torch.float32)

        return latent_tensor, audio_tensor.squeeze(0), class_tensor



class DiffusionTrainer:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=2e-2,
                 device="cuda", prediction_type="epsilon",
                 learn_variance=False, vlb_loss_weight=0.001):
        """
        Diffusion Trainer with v-prediction and learned variance.

        """

        self.model = model.to(device)
        self.device = device
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        self.learn_variance = learn_variance
        self.vlb_loss_weight = vlb_loss_weight

        # Linear variance
        self.betas = torch.linspace(beta_start, beta_end, timesteps)

        self.betas = self.betas.to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # v-prediction
        self.sqrt_alphas_cumprod_minus_one = torch.sqrt(self.alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def q_sample(self, x_start, t, noise=None):
        """Add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise, noise

    def get_velocity(self, x_start, noise, t):
        """velocity target for v-prediction"""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x_start

    def predict_x0_from_eps(self, x_t, t, eps):
        """Predict x_0 from x_t and predicted noise"""
        sqrt_recip_alpha_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alpha_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_recip_alpha_t * x_t - sqrt_recipm1_alpha_t * eps

    def predict_x0_from_v(self, x_t, t, v):
        """Predict x_0 from x_t and predicted velocity"""
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * v

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_losses(self, x_start, t, audio_context, note_density, cfg_dropout_prob=0.1, loss_type="mse"):
        noise = torch.randn_like(x_start)
        x_noisy, _ = self.q_sample(x_start, t, noise)

        model_output = self.model(x_noisy, t, audio_context, note_density, cfg_dropout_prob)

        if self.learn_variance:
            # Split: (B, 2C, H, W) -> (B, C, H, W), (B, C, H, W)
            model_mean_output, model_var_values = torch.split(model_output, x_start.shape[1], dim=1)
        else:
            model_mean_output = model_output
            model_var_values = None

        # Calculate Main Loss
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v":
            target = self.get_velocity(x_start, noise, t)

        main_loss = F.mse_loss(model_mean_output, target)

        # Calculate Variance Loss (L_vlb) -> Trains the variance
        if self.learn_variance:
            # Instead of predicting log_var directly, we predict 'v' and interpolate
            # between log(posterior_variance) and log(beta)

            # Map output to [0, 1] range might be helpful, but OpenAI just uses the raw output
            # effectively as the mix factor (sometimes with a sigmoid).
            # OpenAI uses: (v + 1) / 2 to map [-1, 1] output to [0, 1] range
            # we decided to use this trick
            t_batch = t.reshape(-1, 1, 1, 1)
            min_log = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
            max_log = torch.log(self.betas[t]).reshape(-1, 1, 1, 1)

            # The model output is treated as 'v'. We squash it to [0, 1]
            v = (model_var_values + 1) / 2

            # Interpolate
            model_log_variance = v * max_log + (1 - v) * min_log
            model_mean_detached = model_mean_output.detach()

            if self.prediction_type == "epsilon":
                x_0_pred = self.predict_x0_from_eps(x_noisy, t, model_mean_detached)
            elif self.prediction_type == "v":
                x_0_pred = self.predict_x0_from_v(x_noisy, t, model_mean_detached)

            x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

            # Calculate True Posterior (Ground Truth Distribution)
            true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start, x_noisy, t)

            # Calculate Predicted Posterior (Model Distribution)
            # We use the interpolated variance here
            pred_mean, _, _ = self.q_posterior_mean_variance(x_0_pred, x_noisy, t)

            # KL Divergence computation
            kl = 0.5 * (
                - true_log_variance_clipped
                + model_log_variance
                + torch.exp(true_log_variance_clipped - model_log_variance)
                + ((true_mean - pred_mean) ** 2) * torch.exp(-model_log_variance)
                - 1.0
            )

            # Mask out t=0 because posterior variance is 0 (or close to it)
            # OpenAI typically calculates decoder NLL at t=0, but for latent diffusion
            # masking is sufficient to stop the explosion.
            mask = (t > 0).float().reshape(-1, 1, 1, 1)
            kl_loss = (kl * mask).mean()

            # Total Loss
            loss = main_loss + (self.vlb_loss_weight * kl_loss)

            return loss, {'main_loss': main_loss.item(), 'kl_loss': kl_loss.item()}

        return main_loss, {'main_loss': main_loss.item()}


def train(
    model,
    train_loader,
    val_loader=None,
    epochs=100,
    lr=1e-4,
    device="cuda",
    checkpoint_dir="ckpts",
    save_freq=10,
    grad_clip=1.0,
    warmup_steps=1000,
    ema_decay=0.9999,
    loss_type="huber",
    schedule_type="linear",
    prediction_type="epsilon",
    early_stop_patience=None,
    cond_dropout_prob=0.1,
    gradient_accumulation_steps=16,  # 4 * 16 = 64 effective batch size
    scale_factor=0.18215, # according to stable audio
    use_8bit_adam=True,
    log_dir="./logs",
    use_autocast=True
):
    """
    training loop

    Key features:
    - Linear variance schedule (1e-4 to 2e-2, T=1000)
    - No weight decay, following DiT paper
    - EMA decay = 0.9999
    - 8-bit quantization for memory efficiency
    - Gradient accumulation
    - Mixed precision training
    """
    # Setup
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    writer = SummaryWriter(log_dir=f"{log_dir}/dit_{timestamp}")

    model = model.to(device)
    learn_variance = hasattr(model, 'learn_sigma') and model.learn_sigma

    diffusion = DiffusionTrainer(
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        schedule_type=schedule_type,
        prediction_type=prediction_type,
        learn_variance=learn_variance,
        vlb_loss_weight=0.001,  # Weight for variance learning
        device=device
    )

    # Optimizer
    if use_8bit_adam:
        import bitsandbytes as bnb
        print("Using 8-bit AdamW for memory efficiency")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,  # No weight decay as specified
            betas=(0.9, 0.95)
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.0,  # No weight decay
            betas=(0.9, 0.95)
        )

    # Cosine learning rate with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (epochs * len(train_loader) - warmup_steps)
            return 0.5 * (1 + math.cos(progress * math.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA with decay = 0.9999
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay)
    )

    # Mixed precision scaler for autocast
    scaler = torch.amp.GradScaler('cuda') if use_autocast else None

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    config = {
        "epochs": epochs,
        "lr": lr,
        "weight_decay": 0.0,
        "ema_decay": ema_decay,
        "grad_clip": grad_clip,
        "warmup_steps": warmup_steps,
        "loss_type": loss_type,
        "schedule_type": schedule_type,
        "prediction_type": prediction_type,
        "cond_dropout_prob": cond_dropout_prob,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": 4 * gradient_accumulation_steps,
        "use_8bit_adam": use_8bit_adam,
        "use_autocast": use_autocast,
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
        "num_params": num_params,
        "trainable_params": trainable_params,
        "timestamp": timestamp
    }

    with open(f"{checkpoint_dir}/config_{timestamp}.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"{'='*70}")
    print(f"Diffusion Transformer Training - Optimized for Limited Hardware")
    print(f"{'='*70}")
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    print(f"Variance schedule: Linear (β: 1e-4 → 2e-2, T=1000)")
    print(f"Prediction type: {prediction_type}")
    print(f"Loss: {loss_type}")
    print(f"Weight decay: 0.0 (disabled)")
    print(f"EMA decay: {ema_decay}")
    print(f"CFG dropout: {cond_dropout_prob}")
    print(f"Real batch size: 2")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {4 * gradient_accumulation_steps}")
    print(f"8-bit optimizer: {use_8bit_adam}")
    print(f"Mixed precision: {use_autocast}")
    print(f"{'='*70}\n")

    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0

    conditioner = AudioConditioner()
    conditioner.to(device)
    conditioner.eval()

    for epoch in range(epochs):
        model.train()
        train_metrics = {'loss': 0, 'main_loss': 0, 'kl_loss': 0, 'count': 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            x_start, audio, note_density = batch
            x_start = x_start * scale_factor
            x_start = x_start.to(device)
            note_density = note_density.to(device)
            audio = audio.to(device)

            with torch.no_grad():
                audio_list = [a.cpu().numpy() for a in audio]
                inputs = conditioner.process_audio(audio_list, sampling_rate=24000)
                audio_context = conditioner(inputs['input_values'].to(device))

            t = torch.randint(0, diffusion.timesteps, (x_start.shape[0],), device=device).long()

            if use_autocast:
                with torch.amp.autocast('cuda'):
                    loss, loss_dict = diffusion.p_losses(
                        x_start, t,
                        audio_context=audio_context,
                        note_density=note_density,
                        cfg_dropout_prob=cond_dropout_prob,
                        loss_type=loss_type
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
            else:
                loss, loss_dict = diffusion.p_losses(
                    x_start, t,
                    audio_context=audio_context,
                    note_density=note_density,
                    cfg_dropout_prob=cond_dropout_prob,
                    loss_type=loss_type
                )
                loss = loss / gradient_accumulation_steps

            if use_autocast:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation: only step every N batches
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if grad_clip > 0:
                    if use_autocast:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


                if use_autocast:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                ema_model.update_parameters(model)
                global_step += 1

                # tensorboard logs
                if global_step % 100 == 0:
                    writer.add_scalar('train/loss', loss.item() * gradient_accumulation_steps, global_step)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                    if learn_variance:
                        writer.add_scalar('train/kl_loss', loss_dict.get('kl_loss', 0), global_step)
                        writer.add_scalar('train/main_loss', loss_dict.get('main_loss', 0), global_step)

            # Accumulate metrics
            train_metrics['loss'] += loss.item() * gradient_accumulation_steps
            train_metrics['main_loss'] += loss_dict.get('main_loss', 0)
            train_metrics['kl_loss'] += loss_dict.get('kl_loss', 0)
            train_metrics['count'] += 1

            # progress bar
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                pbar_dict = {
                    'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                    'step': global_step,
                }
                if learn_variance and 'kl_loss' in loss_dict:
                    pbar_dict['kl'] = f"{loss_dict['kl_loss']:.4f}"
                pbar.set_postfix(pbar_dict)

        # Handle remaining gradients at end of epoch
        if len(train_loader) % gradient_accumulation_steps != 0:
            if grad_clip > 0:
                if use_autocast:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_autocast:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            ema_model.update_parameters(model)

        avg_train_loss = train_metrics['loss'] / train_metrics['count']
        avg_train_main = train_metrics['main_loss'] / train_metrics['count']
        avg_train_kl = train_metrics['kl_loss'] / train_metrics['count'] if learn_variance else 0

        # Validation
        val_metrics_str = ""
        if val_loader is not None:
            model.eval()
            val_metrics = {'loss': 0, 'main_loss': 0, 'kl_loss': 0, 'count': 0}

            with torch.no_grad():
                for batch in val_loader:
                    x_start, audio, note_density = batch
                    x_start = x_start.to(device)
                    note_density = note_density.to(device)
                    audio = audio.to(device)

                    # Process audio
                    if conditioner is not None:
                        audio_list = [a.cpu().numpy() for a in audio]
                        inputs = conditioner.process_audio(audio_list, sampling_rate=24000)
                        audio_context = conditioner(inputs['input_values'].to(device))
                    else:
                        audio_context = torch.randn(audio.shape[0], 100, 768, device=device)

                    t = torch.randint(0, diffusion.timesteps, (x_start.shape[0],), device=device).long()

                    if use_autocast:
                        with torch.amp.autocast('cuda'):
                            loss, loss_dict = diffusion.p_losses(
                                x_start, t,
                                audio_context=audio_context,
                                note_density=note_density,
                                cfg_dropout_prob=0.0,
                                loss_type=loss_type
                            )
                    else:
                        loss, loss_dict = diffusion.p_losses(
                            x_start, t,
                            audio_context=audio_context,
                            note_density=note_density,
                            cfg_dropout_prob=0.0, # no cfg during validation
                            loss_type=loss_type
                        )

                    val_metrics['loss'] += loss.item()
                    val_metrics['main_loss'] += loss_dict.get('main_loss', 0)
                    val_metrics['kl_loss'] += loss_dict.get('kl_loss', 0)
                    val_metrics['count'] += 1

            avg_val_loss = val_metrics['loss'] / val_metrics['count']
            avg_val_main = val_metrics['main_loss'] / val_metrics['count']
            avg_val_kl = val_metrics['kl_loss'] / val_metrics['count'] if learn_variance else 0

            if learn_variance:
                val_metrics_str = f" | Val: {avg_val_loss:.4f} (main: {avg_val_main:.4f}, kl: {avg_val_kl:.4f})"
            else:
                val_metrics_str = f" | Val: {avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': ema_model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': config
                }, f"{checkpoint_dir}/dit_best_model.pth")
            else:
                patience_counter += 1

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        log_str = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f}"
        if learn_variance:
            log_str += f" (main: {avg_train_main:.4f}, kl: {avg_train_kl:.4f})"
        log_str += f"{val_metrics_str} | LR: {current_lr:.2e}"
        print(log_str)

        # Save checkpoints
        if (epoch + 1) % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, f"{checkpoint_dir}/dit_epoch_{epoch+1}.pth")

            torch.save({
                'epoch': epoch,
                'model_state_dict': ema_model.module.state_dict(),
                'config': config
            }, f"{checkpoint_dir}/dit_ema_epoch_{epoch+1}.pth")

            print(f"Checkpoint saved (regular + EMA)")

        # Early stopping
        if early_stop_patience and patience_counter >= early_stop_patience:
            print(f"\nEarly stopping after {patience_counter} epochs without improvement")
            break

    # Save
    torch.save({
        'epoch': epochs,
        'model_state_dict': ema_model.module.state_dict(),
        'config': config
    }, f"{checkpoint_dir}/final_dit.pth")

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {checkpoint_dir}/final_dit.pth")
    print(f"{'='*70}")

    return ema_model.module



if __name__ == "__main__":
    from models.transformer import Pop2PianoDiT_MMDiT
    from torch.utils.data import random_split

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="torchaudio"
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    LATENT_SIZE = (16, 128)
    AUDIO_DIM = 768
    BATCH_SIZE = 2

    torch.manual_seed(1234)

    dit_model = Pop2PianoDiT_MMDiT(
        input_size=LATENT_SIZE,
        patch_size=2,
        in_channels=4,
        hidden_size=512,
        depth=8,
        num_heads=8,
        audio_dim=AUDIO_DIM,
        learn_sigma=False
    )

    data_dir = "/mnt/ssd-samsung/atiam/projet_ml/roll_tensor/"
    full_dataset = DiffusionDataset(dataset_root=data_dir)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    trained_model = train(
        model=dit_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=3e-4,
        device="cuda",
        checkpoint_dir="ckpts",
        save_freq=10,
        grad_clip=1.0,
        warmup_steps=200,
        ema_decay=0.9999,
        schedule_type="linear",
        prediction_type="v",
        cond_dropout_prob=0.1,
        gradient_accumulation_steps=8,
        use_8bit_adam=False,  # Memory efficient
        use_autocast=True
    )
