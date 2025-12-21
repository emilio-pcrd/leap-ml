import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.blocks import CrossAttentionDiTBlock


def modulate(x, shift, scale):
    """Apply affine transformation: scale * x + shift
    this is for adaln layers"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):

        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds scalar labels (note density) into vector representations.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        # Null token for CFG
        self.null_token = nn.Parameter(torch.zeros(1, hidden_size))

    def forward(self, labels, train=False, cfg_dropout_prob=0.1):
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        labels = labels.float()
        label_emb = self.mlp(labels)

        # CFG: randomly replace with null token during training
        if train and cfg_dropout_prob > 0:
            batch_size = labels.shape[0]
            mask = torch.rand(batch_size, device=labels.device) < cfg_dropout_prob
            mask = mask.unsqueeze(1)  # (B,1)
            label_emb = torch.where(mask, self.null_token.expand(batch_size, -1), label_emb)

        return label_emb


class MMDiTBlock(nn.Module):
    """
    Multimodal Diffusion Transformer Block (following Stable Diffusion 3 architecture).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_audio = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn_img = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )
        self.attn_audio = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=0.0, batch_first=True
        )

        # FFN for image modality
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )

        # FFN for audio modality
        self.norm2_audio = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_audio = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 12 * hidden_size, bias=True)
        )

        # Initialize to identity (AdaLN-Zero)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x_img, x_audio, c):
        # modulation parameters
        mods = self.adaLN_modulation(c).chunk(12, dim=-1)
        shift_msa_img, scale_msa_img, gate_msa_img = mods[0], mods[1], mods[2]
        shift_mlp_img, scale_mlp_img, gate_mlp_img = mods[3], mods[4], mods[5]
        shift_msa_audio, scale_msa_audio, gate_msa_audio = mods[6], mods[7], mods[8]
        shift_mlp_audio, scale_mlp_audio, gate_mlp_audio = mods[9], mods[10], mods[11]

        # Self-Attention with Joint Attention
        x_img_mod = modulate(self.norm1_img(x_img), shift_msa_img, scale_msa_img)
        x_audio_mod = modulate(self.norm1_audio(x_audio), shift_msa_audio, scale_msa_audio)

        x_joint = torch.cat([x_img_mod, x_audio_mod], dim=1)  # (B, N_img + N_audio, D)

        # Separate attention for each modality, but over joint sequence
        # Image attends to [image, audio]
        attn_img_out, _ = self.attn_img(x_img_mod, x_joint, x_joint)
        x_img = x_img + gate_msa_img.unsqueeze(1) * attn_img_out

        # Audio attends to [image, audio]
        attn_audio_out, _ = self.attn_audio(x_audio_mod, x_joint, x_joint)
        x_audio = x_audio + gate_msa_audio.unsqueeze(1) * attn_audio_out

        # FFN
        x_img_mod = modulate(self.norm2_img(x_img), shift_mlp_img, scale_mlp_img)
        x_img = x_img + gate_mlp_img.unsqueeze(1) * self.mlp_img(x_img_mod)
        x_audio_mod = modulate(self.norm2_audio(x_audio), shift_mlp_audio, scale_mlp_audio)
        x_audio = x_audio + gate_mlp_audio.unsqueeze(1) * self.mlp_audio(x_audio_mod)

        return x_img, x_audio


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        # Initialize to zero (AdaLN-Zero)
        nn.init.normal_(self.adaLN_modulation[-1].weight, std=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Pop2PianoDiT_MMDiT(nn.Module):
    def __init__(
        self,
        input_size=(16, 128),
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        audio_dim=768,
        mlp_ratio=4.0,
        learn_sigma=False
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.learn_sigma = learn_sigma
        self.num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)

        # Patch embedding
        self.x_embedder = nn.Linear(
            patch_size * patch_size * in_channels,
            hidden_size,
            bias=True
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        self.audio_proj = nn.Linear(audio_dim, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(hidden_size)

        # MM-DiT blocks
        self.blocks = nn.ModuleList([
            MMDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transf ormer layers with standard ibnitialization
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep and label embedders
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.y_embedder.mlp[0].bias, 0)

        nn.init.zeros_(self.y_embedder.null_token)

        nn.init.zeros_(self.y_embedder.mlp[2].weight)
        nn.init.zeros_(self.y_embedder.mlp[2].bias)

    def patchify(self, x):
        """
        x: (B, C, H, W)
        return: (B, N, patch_size**2 * C) where N = (H/p) * (W/p)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        h, w = H // p, W // p

        x = x.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x):
        """
        x: (B, N, patch_size**2 * C)
        return: (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = self.input_size[0] // p
        w = self.input_size[1] // p
        c = self.out_channels

        x = x.reshape(B, h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(B, c, h * p, w * p)
        return x

    def forward(self, x, t, audio_context, note_density=None, train=False, cfg_dropout_prob=0.1):
        x_img = self.patchify(x)  #(B,N,p*p*C)
        x_img = self.x_embedder(x_img) + self.pos_embed

        x_audio = self.audio_proj(audio_context)
        t_emb = self.t_embedder(t)

        # Embed label (in our case, note density) with CFG
        if note_density is not None:
            y_emb = self.y_embedder(note_density, train=train, cfg_dropout_prob=cfg_dropout_prob)
        else:
            y_emb = self.y_embedder.null_token.expand(x.shape[0], -1)

        # Combine timestep and label for conditioning vector
        # 0.1 can be changed to more, but we found that training
        # was more stable with a regularization here
        c = t_emb + 0.1 * y_emb

        for block in self.blocks:
            x_img, x_audio = block(x_img, x_audio, c)

        # only process image tokens for output
        x_img = self.final_layer(x_img, c)  # (B,N,p*p*C)

        x = self.unpatchify(x_img)  # (B,C,H,W)

        return x



class Pop2PianoDiT_CrossAttn(nn.Module):
    def __init__(
        self,
        input_size=(16, 128),
        patch_size=2,
        in_channels=4,
        hidden_size=384,
        depth=6,
        num_heads=6,
        audio_dim=768,
        dropout=0.1
    ):
        super().__init__()

        self.H, self.W = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.num_patches_h = self.H // patch_size
        self.num_patches_w = self.W // patch_size

        self.x_embed = nn.Linear(patch_size * patch_size * in_channels, hidden_size)

        num_patches = self.num_patches_h * self.num_patches_w
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            CrossAttentionDiTBlock(hidden_size, num_heads, audio_dim, dropout=dropout)
            for _ in range(depth)
        ])

        # final linear layer
        self.norm_out = nn.LayerNorm(hidden_size)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Xavier init for linear layers
        w = self.x_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        nn.init.normal_(self.pos_embed, std=0.02)

        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def patchify(self, x):
        p = self.patch_size
        h = x.shape[2] // p
        w = x.shape[3] // p
        x = x.reshape(shape=(x.shape[0], 4, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(x.shape[0], h * w, p**2 * 4))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = self.num_patches_h
        w = self.num_patches_w
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 4))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], 4, h * p, w * p))
        return x

    def forward(self, x, t, audio_context):
        """
        x: (N, C, H, W)
        t: (N,)
        audio_context: (N, Seq, Audio_Dim)
        """
        # patchify and embed
        x = self.patchify(x)
        x = self.x_embed(x) + self.pos_embed

        # we add t_emb to the input sequence globally.
        t_emb = self.t_embedder(t) # (N, Hidden_Size)
        x = x + t_emb.unsqueeze(1)

        # blocks of the DiT
        for block in self.blocks:
            x = block(x, audio_context)

        # decoding step
        x = self.norm_out(x)
        x = self.final_linear(x)
        x = self.unpatchify(x)

        return x








if __name__ == "__main__":
    model = Pop2PianoDiT_MMDiT(
        input_size=(16, 128),
        patch_size=2,
        in_channels=4,
        audio_dim=768,
        hidden_size=512,
        depth=8,
        num_heads=8,
        learn_sigma=True
    )

    batch_size = 2
    x = torch.randn(batch_size, 4, 16, 128)
    t = torch.randint(0, 1000, (batch_size,))
    audio = torch.randn(batch_size, 430, 768)
    density = torch.randn(batch_size, 1)

    with torch.no_grad():
        out = model(x, t, audio, density, train=False)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
