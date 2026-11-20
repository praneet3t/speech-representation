import numpy as np
import torch

from models.training import probabilities


def evaluate(model, data, device, batch_size):
    return probabilities(model, data, device, batch_size)


@torch.inference_mode()
def representations(model, speech, text, device, batch_size):
    model.eval()
    pairs = zip(speech.split(batch_size), text.split(batch_size), strict=True)
    return np.concatenate([model.fused_features(a.to(device), b.to(device)).cpu().numpy() for a, b in pairs])
