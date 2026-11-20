import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.features import MELS, masked_mean


class SpeechModel(nn.Module):
    embedding_size = 128

    def __init__(self, classes=7):
        super().__init__()
        self.projection = nn.Conv1d(MELS, 48, kernel_size=3, stride=3, padding=1)
        self.recurrent = nn.GRU(48, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(self.embedding_size, 64), nn.GELU(),
                                        nn.Dropout(.25), nn.Linear(64, classes))

    def temporal_features(self, acoustic, lengths):
        valid = torch.arange(acoustic.shape[1], device=acoustic.device)[None, :] < lengths.to(acoustic.device)[:, None]
        acoustic = acoustic.masked_fill(~valid.unsqueeze(-1), 0)
        projected = torch.nn.functional.gelu(self.projection(acoustic.transpose(1, 2)).transpose(1, 2))
        steps = (lengths + 2) // 3
        packed = pack_padded_sequence(projected, steps.cpu(), batch_first=True, enforce_sorted=False)
        pooled, _ = self.recurrent(packed)
        frames, _ = pad_packed_sequence(pooled, batch_first=True)
        mask = torch.arange(frames.shape[1], device=frames.device)[None, :] < steps.to(frames.device)[:, None]
        return masked_mean(frames, mask)

    def forward(self, acoustic, lengths):
        return self.classifier(self.temporal_features(acoustic, lengths))
