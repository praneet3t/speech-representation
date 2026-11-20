from models.fusion_pipeline.model import FusionModel, SpeechControl
from models.training import fit, seed_randomness


def train(train_data, validation, directory, settings, device, frozen, control=False):
    seed_randomness(settings["seed"])
    model = SpeechControl() if control else FusionModel()
    history, stats = fit(model, train_data, validation, directory, settings, device, frozen)
    return model, history, stats
