"""
Conditional Flow Matching for TTS
Based on F5-TTS and Voicebox approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 block for feature refinement"""
    def __init__(self, dim, intermediate_dim=None, kernel_size=7, drop_path=0.):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = dim * 4

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x):
        """
        x: (B, C, T)
        """
        residual = x
        x = self.dwconv(x)  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = residual + x
        return x


class DurationPredictor(nn.Module):
    """Predicts duration/alignment for text-to-mel"""
    def __init__(self, in_channels, filter_channels=256, kernel_size=3,
                 p_dropout=0.1, n_layers=3):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        # Input projection
        self.conv_layers.append(nn.Conv1d(in_channels, filter_channels,
                                         kernel_size, padding=kernel_size//2))
        self.norm_layers.append(nn.LayerNorm(filter_channels))

        # Hidden layers
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(filter_channels, filter_channels,
                                             kernel_size, padding=kernel_size//2))
            self.norm_layers.append(nn.LayerNorm(filter_channels))

        # Output projection (log duration)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask):
        """
        x: (B, C, T_text)
        x_mask: (B, 1, T_text)
        Returns: (B, T_text) - log duration for each text token
        """
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = conv(x * x_mask)
            x = x.transpose(1, 2)
            x = norm(x)
            x = x.transpose(1, 2)
            x = F.relu(x)
            x = self.drop(x)

        x = self.proj(x * x_mask)  # (B, 1, T_text)
        return x.squeeze(1)  # (B, T_text)


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) timestep values in [0, 1]
        Returns: (B, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FlowMatchingTransformer(nn.Module):
    """
    Flow Matching Transformer for mel-spectrogram generation
    Predicts velocity field v_t for flow matching
    """
    def __init__(self,
                 n_mel_channels=80,
                 n_text_vocab=200,
                 d_model=512,
                 nhead=8,
                 num_layers=12,
                 dim_feedforward=2048,
                 dropout=0.1,
                 use_convnext=True,
                 n_convnext_layers=4):
        super().__init__()

        self.n_mel_channels = n_mel_channels
        self.d_model = d_model
        self.use_convnext = use_convnext

        # Text encoder
        self.text_emb = nn.Embedding(n_text_vocab, d_model)
        self.text_pos_enc = nn.Parameter(torch.randn(1, 5000, d_model) * 0.02)

        # Mel encoder (for noisy input x_t)
        self.mel_proj = nn.Conv1d(n_mel_channels, d_model, 1)

        # Timestep embedding
        self.time_emb = TimestepEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

        # ConvNeXt blocks for text refinement (F5-TTS style)
        if use_convnext:
            self.convnext_blocks = nn.ModuleList([
                ConvNeXtBlock(d_model) for _ in range(n_convnext_layers)
            ])

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection (velocity field)
        self.output_proj = nn.Conv1d(d_model, n_mel_channels, 1)

    def forward(self, x_t, t, text, text_lengths, mel_lengths):
        """
        x_t: (B, n_mel, T_mel) - noisy mel at time t
        t: (B,) - timestep in [0, 1]
        text: (B, T_text) - text token indices
        text_lengths: (B,) - actual text lengths
        mel_lengths: (B,) - actual mel lengths

        Returns: v_t (B, n_mel, T_mel) - predicted velocity field
        """
        B, _, T_mel = x_t.shape
        T_text = text.size(1)

        # Text encoding
        text_emb = self.text_emb(text)  # (B, T_text, d_model)
        text_emb = text_emb + self.text_pos_enc[:, :T_text, :]

        # Apply ConvNeXt refinement to text
        if self.use_convnext:
            text_feat = text_emb.transpose(1, 2)  # (B, d_model, T_text)
            for block in self.convnext_blocks:
                text_feat = block(text_feat)
            text_emb = text_feat.transpose(1, 2)  # (B, T_text, d_model)

        # Mel encoding
        mel_emb = self.mel_proj(x_t)  # (B, d_model, T_mel)
        mel_emb = mel_emb.transpose(1, 2)  # (B, T_mel, d_model)

        # Timestep embedding
        t_emb = self.time_emb(t)  # (B, d_model)
        t_emb = self.time_mlp(t_emb)  # (B, d_model)

        # Add timestep to both text and mel
        text_emb = text_emb + t_emb.unsqueeze(1)
        mel_emb = mel_emb + t_emb.unsqueeze(1)

        # Concatenate text and mel
        combined = torch.cat([text_emb, mel_emb], dim=1)  # (B, T_text+T_mel, d_model)

        # Create attention mask
        max_len = T_text + T_mel
        mask = torch.zeros(B, max_len, device=x_t.device, dtype=torch.bool)
        for i in range(B):
            mask[i, text_lengths[i]:T_text] = True  # Mask padded text
            mask[i, T_text + mel_lengths[i]:] = True  # Mask padded mel

        # Transformer
        output = self.transformer(combined, src_key_padding_mask=mask)  # (B, T_text+T_mel, d_model)

        # Extract mel part
        mel_output = output[:, T_text:, :]  # (B, T_mel, d_model)
        mel_output = mel_output.transpose(1, 2)  # (B, d_model, T_mel)

        # Predict velocity
        v_t = self.output_proj(mel_output)  # (B, n_mel, T_mel)

        return v_t


class ConditionalFlowMatching(nn.Module):
    """
    Conditional Flow Matching for mel-spectrogram generation
    Based on optimal transport formulation
    """
    def __init__(self, model, sigma_min=1e-4):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def compute_conditional_flow(self, x_0, x_1, t):
        """
        Compute conditional flow from x_0 (noise) to x_1 (target)

        x_0: (B, C, T) - source (Gaussian noise)
        x_1: (B, C, T) - target (real mel)
        t: (B,) - timestep in [0, 1]

        Returns:
            x_t: (B, C, T) - interpolated sample
            u_t: (B, C, T) - conditional velocity (ground truth)
        """
        t_expand = t.view(-1, 1, 1)  # (B, 1, 1)

        # Optimal transport conditional flow: x_t = t*x_1 + (1-t)*x_0
        x_t = t_expand * x_1 + (1 - t_expand) * x_0

        # Conditional velocity: u_t = x_1 - x_0
        u_t = x_1 - x_0

        return x_t, u_t

    def forward(self, x_1, text, text_lengths, mel_lengths):
        """
        Training forward pass

        x_1: (B, n_mel, T_mel) - target mel-spectrogram
        text: (B, T_text) - text tokens
        text_lengths: (B,) - actual text lengths
        mel_lengths: (B,) - actual mel lengths

        Returns: loss
        """
        B = x_1.size(0)
        device = x_1.device

        # Sample random timestep
        t = torch.rand(B, device=device)

        # Sample noise
        x_0 = torch.randn_like(x_1)

        # Compute conditional flow
        x_t, u_t = self.compute_conditional_flow(x_0, x_1, t)

        # Predict velocity
        v_t = self.model(x_t, t, text, text_lengths, mel_lengths)

        # Flow matching loss: MSE between predicted and true velocity
        loss = F.mse_loss(v_t, u_t, reduction='none')

        # Apply mask to ignore padded regions
        mask = torch.zeros_like(x_1)
        for i in range(B):
            mask[i, :, :mel_lengths[i]] = 1.0

        loss = (loss * mask).sum() / mask.sum()

        return loss

    @torch.no_grad()
    def sample(self, text, text_lengths, mel_lengths, n_timesteps=10,
               method='euler', cfg_strength=1.0):
        """
        Generate mel-spectrogram using ODE solver

        text: (B, T_text) - text tokens
        text_lengths: (B,) - actual text lengths
        mel_lengths: (B,) - target mel lengths
        n_timesteps: number of ODE steps
        method: 'euler' or 'midpoint'
        cfg_strength: classifier-free guidance strength (not implemented yet)

        Returns: x_1 (B, n_mel, T_mel) - generated mel
        """
        B = text.size(0)
        T_mel = mel_lengths.max().item()
        device = text.device

        # Start from noise
        x_t = torch.randn(B, self.model.n_mel_channels, T_mel, device=device)

        # ODE integration from t=0 to t=1
        dt = 1.0 / n_timesteps

        for step in range(n_timesteps):
            t = torch.full((B,), step * dt, device=device)

            if method == 'euler':
                # Euler method: x_{t+dt} = x_t + dt * v_t
                v_t = self.model(x_t, t, text, text_lengths, mel_lengths)
                x_t = x_t + dt * v_t

            elif method == 'midpoint':
                # Midpoint method (RK2)
                v_t = self.model(x_t, t, text, text_lengths, mel_lengths)
                x_mid = x_t + (dt / 2) * v_t
                t_mid = torch.full((B,), (step + 0.5) * dt, device=device)
                v_mid = self.model(x_mid, t_mid, text, text_lengths, mel_lengths)
                x_t = x_t + dt * v_mid
            else:
                raise ValueError(f"Unknown ODE method: {method}")

        return x_t

    @torch.no_grad()
    def sample_with_sway(self, text, text_lengths, mel_lengths, n_timesteps=10,
                         sway_coef=0.0):
        """
        Sample with Sway Sampling (F5-TTS technique)
        Modifies the inference trajectory for better quality

        sway_coef: typically -1.0, shifts trajectory toward cleaner generation
        """
        B = text.size(0)
        T_mel = mel_lengths.max().item()
        device = text.device

        x_t = torch.randn(B, self.model.n_mel_channels, T_mel, device=device)

        dt = 1.0 / n_timesteps

        for step in range(n_timesteps):
            # Modified timestep with sway
            t_raw = step * dt
            t_sway = t_raw + sway_coef * (1 - t_raw) * t_raw
            t_sway = max(0.0, min(1.0, t_sway))  # Clamp to [0, 1]

            t = torch.full((B,), t_sway, device=device)

            v_t = self.model(x_t, t, text, text_lengths, mel_lengths)
            x_t = x_t + dt * v_t

        return x_t


class FlowMatchingTTS(nn.Module):
    """
    Complete Flow Matching TTS system
    """
    def __init__(self,
                 n_mel_channels=80,
                 n_text_vocab=200,
                 d_model=512,
                 nhead=8,
                 num_layers=12,
                 dim_feedforward=2048,
                 dropout=0.1,
                 use_duration_predictor=True):
        super().__init__()

        # Flow matching model
        flow_model = FlowMatchingTransformer(
            n_mel_channels=n_mel_channels,
            n_text_vocab=n_text_vocab,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.flow_matching = ConditionalFlowMatching(flow_model)

        # Duration predictor (optional, can use fixed alignment or external aligner)
        self.use_duration_predictor = use_duration_predictor
        if use_duration_predictor:
            self.text_encoder = nn.Sequential(
                nn.Embedding(n_text_vocab, d_model),
                nn.Conv1d(d_model, d_model, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, 3, padding=1),
            )
            self.duration_predictor = DurationPredictor(d_model)

    def forward(self, text, text_lengths, mel, mel_lengths):
        """
        Training forward

        Returns: dict with 'flow_loss' and optionally 'duration_loss'
        """
        losses = {}

        # Flow matching loss
        flow_loss = self.flow_matching(mel, text, text_lengths, mel_lengths)
        losses['flow_loss'] = flow_loss

        # Duration prediction loss (optional)
        if self.use_duration_predictor:
            # Create text features
            text_emb = self.text_encoder[0](text).transpose(1, 2)  # (B, d_model, T)
            for layer in self.text_encoder[1:]:
                text_emb = layer(text_emb)

            # Create mask
            text_mask = torch.zeros(text.size(0), 1, text.size(1), device=text.device)
            for i, length in enumerate(text_lengths):
                text_mask[i, :, :length] = 1.0

            # Predict durations (log scale)
            log_duration_pred = self.duration_predictor(text_emb, text_mask)

            # Compute true durations from alignment (simplified: uniform for now)
            # In practice, you'd use MFA or similar for ground truth
            duration_target = torch.zeros_like(log_duration_pred)
            for i in range(text.size(0)):
                avg_duration = mel_lengths[i].float() / text_lengths[i].float()
                duration_target[i, :text_lengths[i]] = torch.log(avg_duration + 1e-8)

            duration_loss = F.mse_loss(log_duration_pred, duration_target, reduction='none')
            duration_loss = (duration_loss * text_mask.squeeze(1)).sum() / text_mask.sum()
            losses['duration_loss'] = duration_loss

        return losses

    @torch.no_grad()
    def infer(self, text, text_lengths, n_timesteps=10, method='euler',
              duration_scale=1.0, sway_coef=-1.0):
        """
        Inference

        text: (B, T_text)
        text_lengths: (B,)
        n_timesteps: number of ODE steps (10-20 for good quality)
        method: 'euler' or 'midpoint'
        duration_scale: scale predicted durations
        sway_coef: Sway sampling coefficient (F5-TTS: -1.0)

        Returns: mel (B, n_mel, T_mel)
        """
        B = text.size(0)
        device = text.device

        # Predict mel lengths
        if self.use_duration_predictor:
            text_emb = self.text_encoder[0](text).transpose(1, 2)
            for layer in self.text_encoder[1:]:
                text_emb = layer(text_emb)

            text_mask = torch.zeros(B, 1, text.size(1), device=device)
            for i, length in enumerate(text_lengths):
                text_mask[i, :, :length] = 1.0

            log_duration_pred = self.duration_predictor(text_emb, text_mask)
            durations = torch.exp(log_duration_pred) * duration_scale
            durations = torch.clamp(durations, min=1.0)

            mel_lengths = torch.zeros(B, dtype=torch.long, device=device)
            for i in range(B):
                mel_lengths[i] = durations[i, :text_lengths[i]].sum().long()
        else:
            # Fixed ratio (need to specify externally)
            mel_lengths = text_lengths * 5  # Typical text:mel ratio

        # Generate mel
        if sway_coef != 0:
            mel = self.flow_matching.sample_with_sway(
                text, text_lengths, mel_lengths,
                n_timesteps=n_timesteps, sway_coef=sway_coef
            )
        else:
            mel = self.flow_matching.sample(
                text, text_lengths, mel_lengths,
                n_timesteps=n_timesteps, method=method
            )

        return mel, mel_lengths


if __name__ == "__main__":
    # Test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = FlowMatchingTTS(
        n_mel_channels=80,
        n_text_vocab=200,
        d_model=512,
        nhead=8,
        num_layers=6,  # Smaller for testing
        dim_feedforward=2048,
        dropout=0.1,
        use_duration_predictor=True
    ).to(device)

    # Dummy data
    B, T_text, T_mel = 2, 20, 100
    text = torch.randint(0, 200, (B, T_text)).to(device)
    text_lengths = torch.tensor([15, 20]).to(device)
    mel = torch.randn(B, 80, T_mel).to(device)
    mel_lengths = torch.tensor([80, 100]).to(device)

    # Training
    losses = model(text, text_lengths, mel, mel_lengths)
    print("Losses:", losses)

    # Inference
    mel_gen, mel_lengths_gen = model.infer(text, text_lengths, n_timesteps=10)
    print(f"Generated mel shape: {mel_gen.shape}, lengths: {mel_lengths_gen}")

    print("✅ Flow Matching TTS test passed!")
