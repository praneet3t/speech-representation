import numpy as np
import torch

from models.training import iter_batches, probabilities


def evaluate(model, data, device, batch_size):
    return probabilities(model, data, device, batch_size)


@torch.inference_mode()
def embeddings(model, data, device, batch_size):
    model.eval()
    model.requires_grad_(False)
    return np.concatenate([model.temporal_features(x.to(device), length).cpu().numpy()
                           for x, length, _ in iter_batches(data, batch_size)])
