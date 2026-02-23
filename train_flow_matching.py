"""
Training script for Flow Matching TTS
Non-autoregressive, fast inference variant
"""

import os
import json
import argparse
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import TextMelLoader, TextMelCollate
from models import FlowMatchingSynthesizer, MultiPeriodDiscriminator

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  assert torch.cuda.is_available(), "CPU training is not allowed."

  hps = utils.get_hparams()

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(hps.train.port)

  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  # Create dataset - using mel-spectrogram instead of audio tokens
  train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=n_gpus,
    rank=rank,
    shuffle=True)
  collate_fn = TextMelCollate(return_ids=False)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
    batch_size=hps.train.batch_size, pin_memory=True,
    collate_fn=collate_fn, sampler=train_sampler)

  if rank == 0:
    eval_dataset = TextMelLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=1, shuffle=False,
      batch_size=1, pin_memory=False, collate_fn=collate_fn)

  # Initialize model
  net_g = FlowMatchingSynthesizer(
    n_text_vocab=hps.model.n_text_vocab,
    n_mel_channels=hps.data.n_mel_channels,
    inter_channels=hps.model.inter_channels,
    d_model=getattr(hps.model, 'flow_d_model', 512),
    nhead=getattr(hps.model, 'flow_nhead', 8),
    num_layers=getattr(hps.model, 'flow_num_layers', 12),
    dim_feedforward=getattr(hps.model, 'flow_dim_feedforward', 2048),
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
    use_duration_predictor=getattr(hps.model, 'use_duration_predictor', True),
    gin_channels=hps.model.gin_channels,
    # Audio params for mel reconstruction
    sampling_rate=hps.data.sampling_rate,
    filter_length=hps.data.filter_length,
    hop_length=hps.data.hop_length,
    win_length=hps.data.win_length,
    mel_fmin=hps.data.mel_fmin,
    mel_fmax=hps.data.mel_fmax,
  ).cuda(rank)

  # Discriminator for vocoder part (optional, for better audio quality)
  use_discriminator = getattr(hps.train, 'use_discriminator', True)
  if use_discriminator:
    net_d = MultiPeriodDiscriminator().cuda(rank)

  # Optimizer
  optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps)

  if use_discriminator:
    optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)

  # DDP
  net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
  if use_discriminator:
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  # Load checkpoint
  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    if use_discriminator:
      _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  # Learning rate scheduler
  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  if use_discriminator:
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  # Mixed precision training
  scaler = GradScaler(enabled=hps.train.fp16_run)

  # Training loop
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank == 0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d] if use_discriminator else [net_g],
                        [optim_g, optim_d] if use_discriminator else [optim_g],
                        [scheduler_g, scheduler_d] if use_discriminator else [scheduler_g],
                        scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d] if use_discriminator else [net_g],
                        [optim_g, optim_d] if use_discriminator else [optim_g],
                        [scheduler_g, scheduler_d] if use_discriminator else [scheduler_g],
                        scaler, [train_loader, None], None, None)
    scheduler_g.step()
    if use_discriminator:
      scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  use_discriminator = len(nets) > 1

  net_g = nets[0]
  if use_discriminator:
    net_d = nets[1]

  optim_g = optims[0]
  if use_discriminator:
    optim_d = optims[1]

  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  global global_step

  net_g.train()
  if use_discriminator:
    net_d.train()

  for batch_idx, batch in enumerate(train_loader):
    text, text_lengths, mel, mel_lengths = batch
    text = text.cuda(rank, non_blocking=True)
    text_lengths = text_lengths.cuda(rank, non_blocking=True)
    mel = mel.cuda(rank, non_blocking=True)
    mel_lengths = mel_lengths.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      # Generator forward
      loss_dict = net_g(text, text_lengths, mel, mel_lengths)

      loss_flow = loss_dict['flow_loss']
      loss_dur = loss_dict.get('duration_loss', torch.tensor(0.0).cuda(rank))
      loss_mel_recon = loss_dict.get('mel_recon_loss', torch.tensor(0.0).cuda(rank))

      loss_gen = loss_flow + 0.1 * loss_dur + 45.0 * loss_mel_recon  # Add mel reconstruction loss

    # Generator backward
    optim_g.zero_grad()
    scaler.scale(loss_gen).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)

    # Discriminator training (optional, for better vocoder)
    if use_discriminator and global_step % 2 == 0:  # Train D less frequently
      with autocast(enabled=hps.train.fp16_run):
        # Generate mel
        with torch.no_grad():
          mel_gen = net_g.module.flow_tts.flow_matching.sample(
            text, text_lengths, mel_lengths, n_timesteps=5)  # Fast generation for training

        # Generate audio from both real and fake mel
        y_hat, _, _ = net_g.module.dec(mel_gen)
        y, _, _ = net_g.module.dec(mel)

        # Handle MB-iSTFT 4D output: (B, subbands, 1, T) -> (B, 1, T)
        if y_hat.dim() == 4:
          y_hat = y_hat.flatten(1, 2)  # (B, subbands * 1, T) = (B, subbands, T)
          y = y.flatten(1, 2)
          # Average across subbands for discriminator
          y_hat = y_hat.mean(dim=1, keepdim=True)  # (B, 1, T)
          y = y.mean(dim=1, keepdim=True)

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

        loss_disc = 0
        for dr, dg in zip(y_d_hat_r, y_d_hat_g):
          loss_disc += torch.mean((dr - 1)**2) + torch.mean(dg**2)

      optim_d.zero_grad()
      scaler.scale(loss_disc).backward()
      scaler.unscale_(optim_d)
      grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
      scaler.step(optim_d)

    scaler.update()

    # Logging
    if rank == 0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses_str = f"[{global_step}] loss_flow={loss_flow:.3f}, loss_dur={loss_dur:.3f}, loss_mel_recon={loss_mel_recon:.3f}"
        if use_discriminator and global_step % 2 == 0:
          losses_str += f", loss_disc={loss_disc:.3f}"
        logger.info(losses_str)

        scalar_dict = {"loss/flow": loss_flow, "loss/dur": loss_dur, "loss/mel_recon": loss_mel_recon,
                      "learning_rate": lr, "grad_norm_g": grad_norm_g}
        if use_discriminator and global_step % 2 == 0:
          scalar_dict["loss/disc"] = loss_disc
          scalar_dict["grad_norm_d"] = grad_norm_d

        utils.summarize(
          writer=writer,
          global_step=global_step,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                            os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        if use_discriminator:
          utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))

    global_step += 1

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval):
  generator.eval()
  with torch.no_grad():
    for batch_idx, batch in enumerate(eval_loader):
      if batch_idx > 4:  # Evaluate only first 5 samples
        break

      text, text_lengths, mel, mel_lengths = batch
      text = text.cuda(0)
      text_lengths = text_lengths.cuda(0)
      mel = mel.cuda(0)
      mel_lengths = mel_lengths.cuda(0)

      # Generate with flow matching
      y_hat, _, mel_gen, _ = generator.module.infer(
        text, text_lengths,
        n_timesteps=20,  # More steps for better quality in evaluation
        sway_coef=-1.0
      )

      # Log
      mel_gen = mel_gen.squeeze(0).cpu()
      audio = y_hat.squeeze(0).cpu()

      # Add to tensorboard
      # writer_eval.add_audio(f'gen/audio_{batch_idx}', audio, global_step, hps.data.sampling_rate)
      # writer_eval.add_image(f'gen/mel_{batch_idx}',
      #                      utils.plot_spectrogram_to_numpy(mel_gen.numpy()),
      #                      global_step, dataformats='HWC')

  generator.train()


if __name__ == "__main__":
  main()
