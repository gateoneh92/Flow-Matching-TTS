# Flow Matching TTS - Offline Training Environment
# Based on current venv setup with PyTorch 2.7.1 + CUDA 11.8

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace/Flow-Matching-TTS

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy frozen requirements
COPY requirements_frozen.txt .

# Install PyTorch with CUDA first (from PyTorch index)
RUN pip install --no-cache-dir torch==2.7.1+cu118 torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other packages (excluding torch and torchaudio from requirements)
RUN grep -v "torch==" requirements_frozen.txt | grep -v "torchaudio==" > requirements_no_torch.txt && \
    pip install --no-cache-dir -r requirements_no_torch.txt

# Copy project files (excluding venv, logs, checkpoints)
COPY *.py ./
COPY text/ ./text/
COPY monotonic_align/ ./monotonic_align/
COPY configs/ ./configs/

# Build monotonic_align
RUN cd monotonic_align && python setup.py build_ext --inplace

# Create directories for training
RUN mkdir -p /workspace/Flow-Matching-TTS/logs \
    /workspace/Flow-Matching-TTS/checkpoints \
    /workspace/Flow-Matching-TTS/outputs \
    /workspace/Flow-Matching-TTS/datasets \
    /workspace/Flow-Matching-TTS/filelists

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command - bash for interactive use
CMD ["/bin/bash"]
