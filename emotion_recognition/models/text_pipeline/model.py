from torch import nn


class TextModel(nn.Module):
    def __init__(self, embedding_size=384, classes=7):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(embedding_size, 48), nn.GELU(),
                                        nn.Dropout(.25), nn.Linear(48, classes))

    def forward(self, text_embedding):
        return self.classifier(text_embedding)
