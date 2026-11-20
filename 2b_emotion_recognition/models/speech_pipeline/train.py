from models.speech_pipeline.model import SpeechModel
from models.training import fit, seed_randomness


def train(train_data, validation, directory, settings, device):
    seed_randomness(settings["seed"])
    model = SpeechModel()
    history, stats = fit(model, train_data, validation, directory, settings, device)
    return model, history, stats
