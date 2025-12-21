import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "../")

from models.blocks import ResnetBlock, AttnBlock, normalization, Downsample, Upsample


class PianoRollResNetVAE(nn.Module):
    def __init__(self,
                 input_channels=1,
                 base_channels=64,  # change this to control model size
                 channel_mults=(1, 2, 4), # depth of the model
                 latent_dim=4):
        super().__init__()

        self.encoder_conv_in = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)

        self.encoder_modules = nn.ModuleList()
        in_ch = base_channels

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            self.encoder_modules.append(ResnetBlock(in_ch, out_ch))
            self.encoder_modules.append(ResnetBlock(out_ch, out_ch))

            if i != len(channel_mults) - 1:
                self.encoder_modules.append(Downsample(out_ch))

            in_ch = out_ch

        # Middle
        self.mid_block1 = ResnetBlock(in_ch, in_ch)
        self.mid_attn = AttnBlock(in_ch)
        self.mid_block2 = ResnetBlock(in_ch, in_ch)

        self.norm_out = normalization(in_ch)
        self.conv_out = nn.Conv2d(in_ch, latent_dim * 2, kernel_size=3, padding=1)

        # decode
        self.decoder_conv_in = nn.Conv2d(latent_dim, in_ch, kernel_size=3, padding=1)

        self.mid_block1_dec = ResnetBlock(in_ch, in_ch)
        self.mid_attn_dec = AttnBlock(in_ch)
        self.mid_block2_dec = ResnetBlock(in_ch, in_ch)

        self.decoder_modules = nn.ModuleList()

        reversed_mults = list(reversed(channel_mults))

        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult

            self.decoder_modules.append(ResnetBlock(in_ch, out_ch))
            self.decoder_modules.append(ResnetBlock(out_ch, out_ch))

            # Upsample everywhere except the last layer
            if i != len(reversed_mults) - 1:
                self.decoder_modules.append(Upsample(out_ch))

            in_ch = out_ch

        self.norm_out_dec = normalization(in_ch)
        self.final_conv = nn.Conv2d(in_ch, input_channels, kernel_size=3, padding=1)

    def encode(self, x):
        h = self.encoder_conv_in(x)
        for m in self.encoder_modules:
            h = m(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = self.norm_out(h)
        h = F.silu(h)
        moments = self.conv_out(h)

        # Split into mean and log_variance
        mu, log_var = torch.chunk(moments, 2, dim=1)
        return mu, log_var

    def decode(self, z):
        h = self.decoder_conv_in(z)

        # Middle
        h = self.mid_block1_dec(h)
        h = self.mid_attn_dec(h)
        h = self.mid_block2_dec(h)

        for m in self.decoder_modules:
            h = m(h)

        h = self.norm_out_dec(h)
        h = F.silu(h)
        h = self.final_conv(h)
        # return torch.sigmoid(h) # binary piano roll
        return h

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model_5m = PianoRollResNetVAE(base_channels=32, channel_mults=(1, 2, 4, 8))
    print(f"Model Parameters: {count_parameters(model_5m):,}")

    dummy_input = torch.randn(2, 1, 128, 1024)
    recon, mu, logvar = model_5m(dummy_input)
    latents = model_5m.reparameterize(*model_5m.encode(dummy_input))
    print(f"Latent Shape: {latents.shape}")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Latent Shape: {mu.shape}")
    print(f"Recon Shape: {recon.shape}")
