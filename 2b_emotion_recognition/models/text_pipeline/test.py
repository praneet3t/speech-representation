from models.training import probabilities


def evaluate(model, data, device, batch_size):
    return probabilities(model, data, device, batch_size)
