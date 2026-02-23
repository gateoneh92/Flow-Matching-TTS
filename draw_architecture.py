"""
Draw Flow Matching TTS Architecture for Paper
Creates a publication-quality diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'stix'

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme (professional, paper-friendly)
color_input = '#E8F4F8'      # Light blue
color_encoder = '#B8E6F0'    # Blue
color_flow = '#FFE5CC'       # Orange
color_duration = '#D4E6D4'   # Green
color_vocoder = '#F0D4E6'    # Purple
color_output = '#FFE8E8'     # Light red

def draw_box(x, y, w, h, text, color, fontsize=10, style='round', linewidth=2):
    """Draw a fancy box with text"""
    if style == 'round':
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=linewidth)
    else:
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="square,pad=0.1",
                            facecolor=color,
                            edgecolor='black',
                            linewidth=linewidth)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight='bold',
           wrap=True)

def draw_arrow(x1, y1, x2, y2, label='', style='->', color='black', linewidth=2):
    """Draw arrow with optional label"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style,
                          color=color,
                          linewidth=linewidth,
                          mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label,
               fontsize=9, ha='left', va='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# Title
ax.text(7, 9.5, 'Flow Matching TTS Architecture',
       ha='center', va='top', fontsize=16, fontweight='bold')

# Input
draw_box(0.5, 8.5, 2, 0.6, 'Text Input\n"Hello World"', color_input, fontsize=10)

# Text Encoder
draw_box(0.5, 7.2, 2, 0.8, 'Text Encoder\n(Embedding)', color_encoder, fontsize=10)
draw_arrow(1.5, 8.5, 1.5, 8.0)

# Duration Predictor (parallel path)
draw_box(3.5, 7.2, 2, 0.8, 'Duration\nPredictor', color_duration, fontsize=10)
draw_arrow(2.5, 7.6, 3.5, 7.6)

# Length Regulator
draw_box(0.5, 5.8, 2, 0.8, 'Length\nRegulator', color_encoder, fontsize=10)
draw_arrow(1.5, 7.2, 1.5, 6.6)
draw_arrow(4.5, 7.2, 4.5, 6.8, '', style='->', color='black')
draw_arrow(4.5, 6.8, 2.5, 6.2, 'durations', style='->', color='black')

# Text Features
draw_arrow(1.5, 5.8, 1.5, 5.0)

# ==================== Flow Matching Section ====================
# Background box for Flow Matching
flow_bg = FancyBboxPatch((0.2, 1.5), 5.6, 3.2,
                        boxstyle="round,pad=0.15",
                        facecolor='#FFF5E6',
                        edgecolor='#FF8C00',
                        linewidth=3,
                        linestyle='--')
ax.add_patch(flow_bg)
ax.text(3, 4.5, 'Conditional Flow Matching (ODE Solver)',
       ha='center', fontsize=11, fontweight='bold', style='italic')

# Gaussian Noise (t=1)
draw_box(0.5, 3.5, 1.5, 0.6, 'Noise z₁\n𝒩(0,I)', '#FFFFFF', fontsize=9, linewidth=1.5)

# Time embedding
draw_box(0.5, 2.6, 1.5, 0.5, 'Time Embed\nt ∈ [0,1]', '#FFFFFF', fontsize=9, linewidth=1.5)

# Flow Transformer (center piece)
draw_box(2.5, 2.2, 2.5, 1.8, 'Flow Matching\nTransformer\n\n12 Layers\nConvNeXt Blocks\nSway Sampling',
         color_flow, fontsize=10, linewidth=2)

# Text condition
draw_arrow(1.5, 4.7, 3.7, 4.0, 'text cond', style='->', color='blue', linewidth=2)

# Noise input
draw_arrow(2.0, 3.5, 2.5, 3.1, '', style='->', linewidth=2)

# Time input
draw_arrow(2.0, 2.85, 2.5, 2.85, '', style='->', linewidth=2)

# ODE steps
draw_arrow(3.7, 2.2, 3.7, 1.9, 'ODE Steps\n(5-20 steps)', style='->', color='#FF6600', linewidth=2)

# Generated Mel
draw_box(2.5, 1.7, 2.5, 0.6, 'Mel-Spectrogram\nz₀ ~ p(x)', color_flow, fontsize=10, linewidth=2)

# ==================== Vocoder Section ====================
# Arrow to vocoder
draw_arrow(5.0, 2.0, 6.5, 2.0)

# Vocoder background
vocoder_bg = FancyBboxPatch((6.3, 1.2), 7.2, 2.5,
                           boxstyle="round,pad=0.15",
                           facecolor='#F9F0FF',
                           edgecolor='#9933FF',
                           linewidth=3,
                           linestyle='--')
ax.add_patch(vocoder_bg)
ax.text(10, 3.5, 'Multi-Band iSTFT Vocoder',
       ha='center', fontsize=11, fontweight='bold', style='italic')

# Conv Pre
draw_box(6.5, 2.6, 1.3, 0.5, 'Conv1d\n(pre)', color_vocoder, fontsize=9)

# Upsampling blocks
draw_box(8.2, 2.8, 1.3, 0.8, 'Upsample\n×4', color_vocoder, fontsize=9)
draw_box(9.8, 2.8, 1.3, 0.8, 'Upsample\n×4', color_vocoder, fontsize=9)

# ResBlocks
draw_box(8.2, 1.8, 1.3, 0.6, 'ResBlocks\n3,7,11', color_vocoder, fontsize=8)
draw_box(9.8, 1.8, 1.3, 0.6, 'ResBlocks\n3,7,11', color_vocoder, fontsize=8)

# Arrows between upsamples
draw_arrow(7.8, 2.9, 8.2, 2.9)
draw_arrow(9.5, 3.0, 9.8, 3.0)

# ResBlock connections
draw_arrow(8.85, 2.8, 8.85, 2.4, style='->', linewidth=1.5)
draw_arrow(10.45, 2.8, 10.45, 2.4, style='->', linewidth=1.5)

# Conv Post
draw_box(11.5, 2.3, 1.4, 0.6, 'Conv Post\n→ STFT', color_vocoder, fontsize=9)
draw_arrow(11.1, 2.9, 11.5, 2.6)

# iSTFT
draw_box(11.5, 1.4, 1.4, 0.6, 'iSTFT\nInverse', color_vocoder, fontsize=9)
draw_arrow(12.2, 2.3, 12.2, 2.0)

# Output
draw_box(11.5, 0.5, 1.4, 0.6, 'Audio\nWaveform', color_output, fontsize=10, linewidth=2)
draw_arrow(12.2, 1.4, 12.2, 1.1)

# ==================== Legend / Info Box ====================
info_y = 0.3
ax.text(0.5, info_y, '● Training: Conditional Flow Matching (Optimal Transport)',
       fontsize=9, va='bottom')
ax.text(0.5, info_y - 0.3, '● Inference: ODE Solver (Euler/Midpoint, 5-20 steps)',
       fontsize=9, va='bottom')
ax.text(0.5, info_y - 0.6, '● Non-autoregressive: 5-20× faster than AR models',
       fontsize=9, va='bottom')
ax.text(0.5, info_y - 0.9, '● RTF: 0.02-0.08 (Real-Time Factor)',
       fontsize=9, va='bottom')

# Model specs
ax.text(7.5, info_y, '● Mel Channels: 80', fontsize=9, va='bottom')
ax.text(7.5, info_y - 0.3, '● Flow Layers: 12 (d=512, h=8)', fontsize=9, va='bottom')
ax.text(7.5, info_y - 0.6, '● Vocoder: 16× upsampling', fontsize=9, va='bottom')
ax.text(7.5, info_y - 0.9, '● Sample Rate: 22050 Hz', fontsize=9, va='bottom')

plt.tight_layout()
plt.savefig('flow_matching_tts_architecture.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.savefig('flow_matching_tts_architecture.pdf', bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("Architecture diagram saved:")
print("- flow_matching_tts_architecture.png (300 DPI)")
print("- flow_matching_tts_architecture.pdf (vector)")
plt.show()
