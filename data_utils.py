"""
Data utilities for Flow Matching TTS
"""

import os
import random
import torch
import torch.utils.data
import commons
from text import cleaned_text_to_sequence, text_to_sequence


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def load_wav_to_torch(full_path):
    import scipy.io.wavfile as wavfile
    sampling_rate, data = wavfile.read(full_path)
    return torch.FloatTensor(data.astype('float32')), sampling_rate


class TextMelLoader(torch.utils.data.Dataset):
    """
    DataLoader for Flow Matching TTS
    Loads text and mel-spectrogram pairs
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.n_mel_channels = hparams.n_mel_channels
        self.mel_fmin = hparams.mel_fmin
        self.mel_fmax = hparams.mel_fmax

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        import torchaudio
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=self.filter_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_mels=self.n_mel_channels
        )

        self._filter()

    def _filter(self):
        """Filter out data with text length outside min/max range"""
        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        audiopath, text = self.audiopaths_and_text[index]

        # Load text
        text = self.get_text(text)

        # Load audio and compute mel
        audio = self.get_audio(audiopath)
        mel = self.mel_spectrogram(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))  # Log mel
        mel = mel.squeeze(0)  # Remove batch dim

        return text, mel

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """Collate function for TextMelLoader"""
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate batch of (text, mel) pairs"""
        # Right zero-pad all sequences
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(0) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_mel_len = max([x[1].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_mel_len)

        text_padded.zero_()
        mel_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            mel = row[1]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

        if self.return_ids:
            return text_padded, text_lengths, mel_padded, mel_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, mel_padded, mel_lengths
