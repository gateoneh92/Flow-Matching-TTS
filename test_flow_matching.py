"""
Test Flow Matching implementation
Verify all components work correctly
"""

import torch
import sys

def test_flow_matching_core():
    """Test core Flow Matching module"""
    print("=" * 60)
    print("Testing Flow Matching Core Module")
    print("=" * 60)

    from flow_matching import FlowMatchingTTS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = FlowMatchingTTS(
        n_mel_channels=80,
        n_text_vocab=200,
        d_model=256,  # Smaller for testing
        nhead=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        use_duration_predictor=True
    ).to(device)

    print(f"✅ Model created successfully")

    # Dummy data
    B, T_text, T_mel = 2, 20, 100
    text = torch.randint(0, 200, (B, T_text)).to(device)
    text_lengths = torch.tensor([15, 20]).to(device)
    mel = torch.randn(B, 80, T_mel).to(device)
    mel_lengths = torch.tensor([80, 100]).to(device)

    # Test training
    print("\nTesting training forward pass...")
    losses = model(text, text_lengths, mel, mel_lengths)
    print(f"✅ Training losses:")
    for key, value in losses.items():
        print(f"   {key}: {value.item():.4f}")

    # Test inference
    print("\nTesting inference (10 steps)...")
    mel_gen, mel_lengths_gen = model.infer(text, text_lengths, n_timesteps=10)
    print(f"✅ Generated mel shape: {mel_gen.shape}")
    print(f"   Mel lengths: {mel_lengths_gen.tolist()}")

    # Test with Sway sampling
    print("\nTesting inference with Sway sampling...")
    mel_gen_sway, _ = model.infer(text, text_lengths, n_timesteps=10, sway_coef=-1.0)
    print(f"✅ Generated mel (sway) shape: {mel_gen_sway.shape}")

    # Test midpoint method
    print("\nTesting inference with midpoint ODE solver...")
    mel_gen_mid, _ = model.infer(text, text_lengths, n_timesteps=10,
                                 method='midpoint', sway_coef=0.0)
    print(f"✅ Generated mel (midpoint) shape: {mel_gen_mid.shape}")

    print("\n" + "=" * 60)
    print("✅ All Flow Matching core tests passed!")
    print("=" * 60)


def test_flow_matching_synthesizer():
    """Test FlowMatchingSynthesizer (integrated with MB-iSTFT)"""
    print("\n" + "=" * 60)
    print("Testing FlowMatchingSynthesizer (Full Model)")
    print("=" * 60)

    from models import FlowMatchingSynthesizer

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    model = FlowMatchingSynthesizer(
        n_text_vocab=200,
        n_mel_channels=80,
        inter_channels=192,
        d_model=256,  # Smaller for testing
        nhead=4,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.1,
        resblock='1',
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[4, 4],
        upsample_initial_channel=512,
        upsample_kernel_sizes=[16, 16],
        gen_istft_n_fft=16,
        gen_istft_hop_size=4,
        subbands=4,
        use_duration_predictor=True,
        gin_channels=0,
    ).to(device)

    print(f"✅ FlowMatchingSynthesizer created successfully")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {n_params:,}")

    # Dummy data
    B, T_text, T_mel = 2, 20, 100
    text = torch.randint(0, 200, (B, T_text)).to(device)
    text_lengths = torch.tensor([15, 20]).to(device)
    mel = torch.randn(B, 80, T_mel).to(device)
    mel_lengths = torch.tensor([80, 100]).to(device)

    # Test training
    print("\nTesting training forward pass...")
    losses = model(text, text_lengths, mel, mel_lengths)
    print(f"✅ Training losses:")
    for key, value in losses.items():
        print(f"   {key}: {value.item():.4f}")

    # Test inference
    print("\nTesting inference (full pipeline: text → mel → audio)...")
    audio, audio_mb, mel_gen, mel_lengths_gen = model.infer(
        text, text_lengths,
        n_timesteps=5,  # Fast for testing
        sway_coef=-1.0
    )
    print(f"✅ Generated audio shape: {audio.shape}")
    print(f"   Audio multiband: {len(audio_mb)} bands")
    print(f"   Mel shape: {mel_gen.shape}")
    print(f"   Mel lengths: {mel_lengths_gen.tolist()}")

    # Test direct mel-to-audio
    print("\nTesting direct mel-to-audio conversion...")
    audio_from_mel, audio_mb_from_mel = model.infer_with_mel(mel)
    print(f"✅ Audio from mel shape: {audio_from_mel.shape}")

    print("\n" + "=" * 60)
    print("✅ All FlowMatchingSynthesizer tests passed!")
    print("=" * 60)


def test_speed_comparison():
    """Compare inference speed: AR vs Flow Matching"""
    print("\n" + "=" * 60)
    print("Speed Comparison: AR vs Flow Matching")
    print("=" * 60)

    import time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cpu':
        print("⚠️  CPU detected, skipping speed test (use GPU for accurate comparison)")
        return

    from flow_matching import FlowMatchingTTS

    # Create model
    model = FlowMatchingTTS(
        n_mel_channels=80,
        n_text_vocab=200,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ).to(device)

    # Dummy data
    B, T_text = 1, 50
    text = torch.randint(0, 200, (B, T_text)).to(device)
    text_lengths = torch.tensor([50]).to(device)

    # Warmup
    _ = model.infer(text, text_lengths, n_timesteps=5)

    # Test different step counts
    for n_steps in [5, 10, 20]:
        start = time.time()
        for _ in range(10):  # Average over 10 runs
            mel, _ = model.infer(text, text_lengths, n_timesteps=n_steps)
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        mel_len = mel.size(2)
        audio_len = mel_len * 256 / 22050  # hop_length=256, sr=22050
        rtf = elapsed / audio_len

        print(f"Steps={n_steps:2d}: {elapsed:.3f}s, RTF={rtf:.3f}, "
              f"mel_len={mel_len}, audio={audio_len:.2f}s")

    print("\n✅ Speed test completed!")
    print("   Note: RTF < 1.0 means faster than real-time")
    print("=" * 60)


def main():
    """Run all tests"""
    try:
        # Test 1: Core Flow Matching
        test_flow_matching_core()

        # Test 2: Full Synthesizer
        test_flow_matching_synthesizer()

        # Test 3: Speed comparison
        test_speed_comparison()

        print("\n" + "🎉" * 30)
        print("ALL TESTS PASSED! Flow Matching is ready to use.")
        print("🎉" * 30)

    except Exception as e:
        print("\n" + "❌" * 30)
        print(f"TEST FAILED: {e}")
        print("❌" * 30)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
