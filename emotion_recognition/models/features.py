from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


SAMPLE_RATE = 16000
WINDOW = 400
HOP = 160
FFT = 512
MELS = 64
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def prepare_waveform(waveform, rate):
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if rate != SAMPLE_RATE:
        ratio = Fraction(SAMPLE_RATE, rate)
        waveform = resample_poly(waveform, ratio.numerator, ratio.denominator)
    # Keep 100 ms around activity and drop at most 250 ms at each edge.
    width = 160
    blocks = np.pad(waveform, (0, (-len(waveform)) % width)).reshape(-1, width)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    active = np.flatnonzero(rms > max(.001, rms.max(initial=0) * .025))
    if len(active):
        left = min(4000, max(0, (active[0] - 10) * width))
        right = min(len(waveform), max(len(waveform) - 4000, (active[-1] + 11) * width))
        return waveform[left:right]
    return waveform


def mel_bank():
    frequencies = np.fft.rfftfreq(FFT, 1 / SAMPLE_RATE)
    low, high = (2595 * np.log10(1 + value / 700) for value in (30, 7800))
    bands = 700 * (10 ** (np.linspace(low, high, MELS + 2) / 2595) - 1)
    left, middle, right = bands[:-2], bands[1:-1], bands[2:]
    rising = (frequencies[:, None] - left) / (middle - left)
    falling = (right - frequencies[:, None]) / (right - middle)
    return np.maximum(0, np.minimum(rising, falling)).astype(np.float32)


BANK = mel_bank()
TAPER = np.hanning(WINDOW).astype(np.float32)


def log_mel_features(waveform, rate):
    samples = prepare_waveform(waveform, rate)
    samples = np.pad(samples, (0, max(0, WINDOW - len(samples))))
    frames = np.lib.stride_tricks.sliding_window_view(samples, WINDOW)[::HOP]
    power = np.abs(np.fft.rfft(frames * TAPER, n=FFT)) ** 2 / (TAPER @ TAPER)
    return np.log(np.maximum(power @ BANK, 1e-10)).astype(np.float32)


def masked_mean(values, mask):
    mask = mask.unsqueeze(-1)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def cache_text(transcripts, work, device):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    directory = work / "minilm"
    required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.txt", "model.safetensors"]
    if not all((directory / name).is_file() for name in required):
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=directory, allow_patterns=required)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    encoder = AutoModel.from_pretrained(directory, local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    unique = sorted(set(transcripts))
    pooled, contextual, masks = [], [], []
    # A corpus-wide padding length lets us archive the original contextual tensor.
    token_count = max(map(len, tokenizer(unique)["input_ids"]))
    with torch.inference_mode():
        for start in range(0, len(unique), 32):
            inputs = tokenizer(unique[start:start + 32], padding="max_length",
                               max_length=token_count, return_tensors="pt").to(device)
            hidden = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            contextual.append(hidden.cpu().numpy())
            masks.append(mask.cpu().numpy())
            pooled.append(masked_mean(hidden, mask).cpu().numpy())
    vectors = np.concatenate(pooled)
    np.savez_compressed(work / "contextual.npz", text=np.array(unique), pooled=vectors,
                        contextual=np.concatenate(contextual), attention_mask=np.concatenate(masks))
    return {"model": MODEL_ID, "revision": REVISION, "dimensions": int(encoder.config.hidden_size),
            "unique_transcripts": len(unique), "token_count": token_count,
            "frozen_parameters": sum(p.numel() for p in encoder.parameters())}
