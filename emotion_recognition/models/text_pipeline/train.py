from models.text_pipeline.model import TextModel
from models.training import fit, seed_randomness


def train(train_data, validation, directory, settings, device, encoder_size):
    seed_randomness(settings["seed"])
    model = TextModel()
    history, stats = fit(model, train_data, validation, directory, settings, device, encoder_size)
    return model, history, stats
