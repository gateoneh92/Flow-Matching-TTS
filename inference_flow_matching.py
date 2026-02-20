"""
Inference script for Flow Matching TTS
Fast, high-quality speech synthesis
"""

import torch
import argparse
import json
from scipy.io import wavfile
import utils
from models import FlowMatchingSynthesizer
from text import text_to_sequence


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load Flow Matching model from checkpoint"""

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    hps = utils.HParams(**config)

    # Initialize model
    net_g = FlowMatchingSynthesizer(
        n_text_vocab=hps.model.n_text_vocab,
        n_mel_channels=hps.data.n_mel_channels,
        inter_channels=hps.model.inter_channels,
        d_model=hps.model.get('flow_d_model', 512),
        nhead=hps.model.get('flow_nhead', 8),
        num_layers=hps.model.get('flow_num_layers', 12),
        dim_feedforward=hps.model.get('flow_dim_feedforward', 2048),
        dropout=hps.model.p_dropout,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        gen_istft_n_fft=hps.model.gen_istft_n_fft,
        gen_istft_hop_size=hps.model.gen_istft_hop_size,
        subbands=hps.model.subbands,
        use_duration_predictor=hps.model.get('use_duration_predictor', True),
        gin_channels=hps.model.gin_channels,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_g.load_state_dict(checkpoint['model'])
    net_g.eval()

    return net_g, hps


def synthesize(model, text, hps, device='cuda',
               n_timesteps=20, duration_scale=1.0,
               sway_coef=-1.0, method='euler'):
    """
    Synthesize speech from text

    Args:
        model: FlowMatchingSynthesizer model
        text: str, input text
        hps: hyperparameters
        device: 'cuda' or 'cpu'
        n_timesteps: number of ODE steps (10-20 for good quality, 30+ for best)
        duration_scale: >1.0 for slower speech, <1.0 for faster
        sway_coef: Sway sampling coefficient (F5-TTS: -1.0, disabled: 0.0)
        method: 'euler' or 'midpoint' ODE solver

    Returns:
        audio: numpy array, waveform
        sample_rate: int
    """
    # Convert text to sequence
    text_norm = text_to_sequence(text, hps.data.text_cleaners)

    if hps.data.add_blank:
        # Add blank tokens between phonemes
        text_norm = commons.intersperse(text_norm, 0)

    # Prepare input
    text_tensor = torch.LongTensor(text_norm).unsqueeze(0).to(device)
    text_lengths = torch.LongTensor([len(text_norm)]).to(device)

    # Generate
    with torch.no_grad():
        audio, _, mel, mel_lengths = model.infer(
            text_tensor,
            text_lengths,
            n_timesteps=n_timesteps,
            duration_scale=duration_scale,
            sway_coef=sway_coef,
            method=method
        )

    audio = audio.squeeze().cpu().numpy()

    return audio, hps.data.sampling_rate, mel.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output_flow.wav',
                       help='Output wav file path')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of ODE steps (10-30)')
    parser.add_argument('--duration-scale', type=float, default=1.0,
                       help='Duration scale (>1.0 slower, <1.0 faster)')
    parser.add_argument('--sway-coef', type=float, default=-1.0,
                       help='Sway sampling coefficient (F5-TTS: -1.0)')
    parser.add_argument('--method', type=str, default='euler',
                       choices=['euler', 'midpoint'],
                       help='ODE solver method')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, hps = load_model(args.checkpoint, args.config, args.device)

    # Synthesize
    print(f"Synthesizing: {args.text}")
    print(f"Settings: steps={args.steps}, duration_scale={args.duration_scale}, "
          f"sway_coef={args.sway_coef}, method={args.method}")

    audio, sr, mel = synthesize(
        model, args.text, hps, args.device,
        n_timesteps=args.steps,
        duration_scale=args.duration_scale,
        sway_coef=args.sway_coef,
        method=args.method
    )

    # Save
    wavfile.write(args.output, sr, (audio * 32768.0).astype('int16'))
    print(f"✅ Audio saved to {args.output}")
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {len(audio)/sr:.2f} seconds")
    print(f"   Mel shape: {mel.shape}")


if __name__ == "__main__":
    main()


# Example usage:
"""
python inference_flow_matching.py \
    --checkpoint logs/flow_matching/G_100000.pth \
    --config configs/flow_matching.json \
    --text "Hello world, this is a test of flow matching TTS." \
    --output output_flow.wav \
    --steps 20 \
    --sway-coef -1.0

# Fast inference (10 steps)
python inference_flow_matching.py \
    --checkpoint logs/flow_matching/G_100000.pth \
    --config configs/flow_matching.json \
    --text "Quick test with only ten steps." \
    --output output_fast.wav \
    --steps 10

# High quality (30 steps)
python inference_flow_matching.py \
    --checkpoint logs/flow_matching/G_100000.pth \
    --config configs/flow_matching.json \
    --text "Highest quality with thirty steps." \
    --output output_hq.wav \
    --steps 30 \
    --method midpoint
"""
