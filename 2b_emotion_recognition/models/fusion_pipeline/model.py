import torch
from torch import nn


class FusionModel(nn.Module):
    def __init__(self, speech_size=128, text_size=384, classes=7):
        super().__init__()
        self.speech = nn.Sequential(nn.Linear(speech_size, 48), nn.GELU())
        self.text = nn.Sequential(nn.Linear(text_size, 32), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(80, 64), nn.GELU())
        self.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(64, classes))

    def fused_features(self, speech_embedding, text_embedding):
        return self.fusion(torch.cat((self.speech(speech_embedding), self.text(text_embedding)), dim=1))

    def forward(self, speech_embedding, text_embedding):
        return self.classifier(self.fused_features(speech_embedding, text_embedding))


class SpeechControl(nn.Module):
    def __init__(self, speech_size=128, classes=7):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(speech_size, 178), nn.GELU(),
                                        nn.Dropout(.25), nn.Linear(178, classes))

    def forward(self, speech_embedding):
        return self.classifier(speech_embedding)
