import json
import os
import random
import resource
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from models.dataset import EMOTIONS, write_json


def seed_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_acoustic_normalizer(features, indices):
    frames = np.concatenate([features[i] for i in indices])
    return frames.mean(axis=0), np.maximum(frames.std(axis=0), 1e-5)


def acoustic_batch(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    return pad_sequence(sequences, batch_first=True), lengths, torch.tensor(labels)


class AcousticData:
    def __init__(self, features, labels, indices, normalizer):
        mean, scale = normalizer
        self.inputs = [torch.from_numpy((features[i] - mean) / scale) for i in indices]
        self.labels = torch.as_tensor(labels[indices], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]


def iter_batches(data, batch_size, shuffle=False, seed=0):
    collate = acoustic_batch if isinstance(data, AcousticData) else None
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=collate)


def fixed_data(arrays, labels, indices):
    return TensorDataset(*(torch.as_tensor(x[indices], dtype=torch.float32) for x in arrays),
                         torch.as_tensor(labels[indices], dtype=torch.long))


def epoch(model, data, device, batch_size, optimizer=None, seed=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.
    true, predicted = [], []
    with torch.set_grad_enabled(training):
        for batch in iter_batches(data, batch_size, shuffle=training, seed=seed):
            features, labels = batch[:-1], batch[-1].to(device)
            args = [x if x.dtype == torch.long else x.to(device) for x in features]
            logits = model(*args)
            loss = nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            true.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    true, predicted = np.asarray(true), np.asarray(predicted)
    return {"loss": total_loss / len(data), "accuracy": float(np.mean(true == predicted)),
            "macro_f1": f1_score(true, predicted, labels=np.arange(len(EMOTIONS)),
                                 average="macro", zero_division=0)}


def atomic_save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def fit(model, train, validation, directory, settings, device, frozen_parameters=0):
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "settings.json"
    settings = {**settings, "device": str(device), "frozen_parameters": frozen_parameters}
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError(f"Checkpoint settings differ: {directory}")
    write_json(config_path, settings)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history, best, bad_epochs, elapsed = [], None, 0, 0.
    latest = directory / "last.pt"
    completed = False
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, best = saved["history"], saved["best"]
        bad_epochs, elapsed, completed = saved["bad_epochs"], saved["seconds"], saved["completed"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for number in range(len(history) + 1, settings["epochs"] + 1):
        if completed:
            break
        started = time.perf_counter()
        train_stats = epoch(model, train, device, settings["batch"], optimizer, settings["seed"] + number)
        val_stats = epoch(model, validation, device, settings["batch"])
        score = (val_stats["macro_f1"], -val_stats["loss"])
        history.append({"epoch": number, "learning_rate": optimizer.param_groups[0]["lr"],
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"val_{k}": v for k, v in val_stats.items()}})
        if best is None or score > tuple(best["score"]):
            best = {"score": score, "epoch": number,
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        elapsed += time.perf_counter() - started
        completed = bad_epochs >= settings["patience"] or number == settings["epochs"]
        atomic_save(latest, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "history": history, "best": best, "bad_epochs": bad_epochs,
                             "seconds": elapsed, "completed": completed, "rng": torch.get_rng_state(),
                             "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        if number == 1 or number % 5 == 0 or completed:
            print(f"{directory.name} epoch {number:2}: val F1 {val_stats['macro_f1']:.3f} "
                  f"best {best['epoch']}  {elapsed:.1f}s", flush=True)
    model.load_state_dict(best["model"])
    model.eval()
    return history, {"checkpoint_epoch": best["epoch"], "epochs_trained": len(history),
                     "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "frozen_parameters": frozen_parameters, "training_seconds": elapsed,
                     "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


@torch.inference_mode()
def probabilities(model, data, device, batch_size):
    model.eval()
    output = []
    started = time.perf_counter()
    for batch in iter_batches(data, batch_size):
        inputs = [x if x.dtype == torch.long else x.to(device) for x in batch[:-1]]
        output.append(model(*inputs).softmax(dim=1).cpu().numpy())
    return np.concatenate(output), time.perf_counter() - started
