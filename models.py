"""
Flow Matching TTS Models
Includes only essential components for Flow Matching
"""

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
from pqmf import PQMF
from stft import TorchSTFT


class Multiband_iSTFT_Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=0):
        super(Multiband_iSTFT_Generator, self).__init__()
        self.subbands = subbands
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3))
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.post_n_fft = gen_istft_n_fft
        self.conv_post = weight_norm(Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = torch.nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.reflection_pad(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:,:self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1:, :])

        return self.stft.inverse(spec, phase).unsqueeze(-2), spec, phase

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class FlowMatchingSynthesizer(nn.Module):
  """
  Flow Matching-based Synthesizer for fast, high-quality TTS
  Combines Flow Matching with MB-iSTFT vocoder
  Non-autoregressive, 5-20x faster than AR models
  """
  def __init__(self,
    n_text_vocab,
    n_mel_channels=80,
    inter_channels=192,
    d_model=512,
    nhead=8,
    num_layers=12,
    dim_feedforward=2048,
    dropout=0.1,
    resblock='1',
    resblock_kernel_sizes=[3,7,11],
    resblock_dilation_sizes=[[1,3,5], [1,3,5], [1,3,5]],
    upsample_rates=[4,4],
    upsample_initial_channel=512,
    upsample_kernel_sizes=[16,16],
    gen_istft_n_fft=16,
    gen_istft_hop_size=4,
    subbands=4,
    use_duration_predictor=True,
    gin_channels=0,
    **kwargs):
    super().__init__()

    # Flow Matching Part
    from flow_matching import FlowMatchingTTS
    self.flow_tts = FlowMatchingTTS(
        n_mel_channels=n_mel_channels,
        n_text_vocab=n_text_vocab,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_duration_predictor=use_duration_predictor
    )

    # Mel-to-Waveform Decoder (MB-iSTFT-VITS)
    self.dec = Multiband_iSTFT_Generator(
        n_mel_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes,
        upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
        gen_istft_n_fft, gen_istft_hop_size, subbands, gin_channels=gin_channels)

    self.n_mel_channels = n_mel_channels

  def forward(self, text, text_lengths, mel, mel_lengths):
    """
    Training forward

    text: (B, T_text) - text token indices
    text_lengths: (B,) - actual text lengths
    mel: (B, n_mel, T_mel) - target mel-spectrogram
    mel_lengths: (B,) - actual mel lengths

    Returns: dict with losses
    """
    losses = self.flow_tts(text, text_lengths, mel, mel_lengths)
    return losses

  def infer(self, text, text_lengths, n_timesteps=10, duration_scale=1.0,
            sway_coef=-1.0, method='euler', g=None, max_len=None):
    """
    Inference - Generate speech from text

    text: (B, T_text) - text token indices
    text_lengths: (B,) - actual text lengths
    n_timesteps: number of ODE steps (10-20 recommended)
    duration_scale: scale predicted durations (>1.0 = slower speech)
    sway_coef: Sway sampling coefficient (-1.0 recommended for F5-TTS style)
    method: 'euler' or 'midpoint' ODE solver

    Returns:
      o: (B, 1, T_audio) - generated waveform
      spec: mel-spectrogram
      phase: phase
      mel_lengths: (B,) - actual mel lengths
    """
    # Generate mel-spectrogram using flow matching
    mel, mel_lengths = self.flow_tts.infer(
        text, text_lengths,
        n_timesteps=n_timesteps,
        duration_scale=duration_scale,
        sway_coef=sway_coef,
        method=method
    )

    # Convert mel to waveform using MB-iSTFT
    if max_len is not None:
      mel = mel[:, :, :max_len]

    o, spec, phase = self.dec(mel, g=g)

    return o, spec, phase, mel_lengths

  def infer_with_mel(self, mel, g=None):
    """
    Direct mel-to-waveform conversion (for testing vocoder separately)

    mel: (B, n_mel, T_mel) - input mel-spectrogram

    Returns:
      o: (B, 1, T_audio) - generated waveform
      spec: mel-spectrogram
      phase: phase
    """
    return self.dec(mel, g=g)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            modules.DiscriminatorP(2),
            modules.DiscriminatorP(3),
            modules.DiscriminatorP(5),
            modules.DiscriminatorP(7),
            modules.DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
