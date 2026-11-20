from importlib import import_module
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project))

from models.dataset import EMOTIONS, linguistic_audit, make_split, parse_filename, transcript, verify_split
from models.features import HOP, MELS, SAMPLE_RATE, WINDOW, log_mel_features, masked_mean
from models.fusion_pipeline.model import FusionModel, SpeechControl
from models.speech_pipeline.model import SpeechModel
from models.speech_pipeline.test import embeddings
from models.text_pipeline.model import TextModel
from models.training import AcousticData, acoustic_batch, fit_acoustic_normalizer

analysis = import_module(f"{project.name}.scripts.analyze")
preparation = import_module(f"{project.name}.scripts.prepare_data")


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def records():
    return [{"id": f"{speaker}-{word}-{emotion}", "pcm_sha256": f"{speaker}-{word}-{emotion}",
             "speaker": speaker, "word": word, "emotion": emotion, "transcript": transcript(word)}
            for speaker, word, emotion in itertools.product(("OAF", "YAF"),
                                                           [f"w{i}" for i in range(20)], EMOTIONS)]


def test_filename_and_transcript_exclude_metadata():
    assert parse_filename("a/YAF_fear/YAF_word_ps.wav") == ("YAF", "word", "surprise")
    assert parse_filename("OAF_neutral/OA_bite_neutral.wav") == ("OAF", "bite", "neutral")
    with pytest.raises(ValueError):
        parse_filename("something_train_label.wav")
    assert transcript("word") == "Say the word word"


def test_duplicate_waveform_checks():
    row = {"id": "one", "pcm_sha256": "alpha"}
    assert preparation.select_unique([row, dict(row)]) == ([row], 1)
    with pytest.raises(ValueError, match="Differing waveforms"):
        preparation.select_unique([row, {"id": "one", "pcm_sha256": "beta"}])
    with pytest.raises(ValueError, match="different IDs"):
        preparation.select_unique([row, {"id": "two", "pcm_sha256": "alpha"}])


@pytest.mark.parametrize("protocol", ["word", "holdout_oaf", "holdout_yaf"])
def test_partitions_are_reproducible_and_disjoint(protocol):
    rows = records()
    a = make_split(rows, protocol)
    assert a == make_split(rows, protocol)
    verify_split(rows, a, protocol)
    groups = {part: [r for r, label in zip(rows, a) if label == part] for part in set(a)}
    if protocol == "word":
        train = {r["word"] for r in groups["train"]}
        assert not train & {r["word"] for r in groups["validation"] + groups["test"]}
    else:
        assert {r["speaker"] for r in groups["train"]} != {r["speaker"] for r in groups["test"]}
        assert {r["speaker"] for r in groups["train"]} == {r["speaker"] for r in groups["validation"]}


def test_partition_validator_rejects_waveform_leakage():
    rows = records()
    split = make_split(rows, "holdout_oaf")
    index = split.index("train")
    rows[index]["pcm_sha256"] = rows[split.index("test")]["pcm_sha256"]
    with pytest.raises(ValueError, match="Leaked pcm_sha256"):
        verify_split(rows, split, "holdout_oaf")


def test_transcript_entropy_and_bound():
    audit = linguistic_audit(records())
    assert audit["unique_transcripts"] == 20
    assert audit["deterministic_text_accuracy_bound"] == pytest.approx(1 / 7)
    assert audit["mutual_information_bits"] == pytest.approx(0, abs=1e-12)


def test_mel_batch_and_training_only_normalization():
    source = np.sin(2 * np.pi * 240 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    mel = log_mel_features(source, SAMPLE_RATE)
    assert mel.shape == ((len(source) - WINDOW) // HOP + 1, MELS)
    assert np.isfinite(mel).all()
    audio, lengths, labels = acoustic_batch([(torch.ones(4, MELS), 2), (torch.ones(2, MELS), 3)])
    assert audio.shape == (2, 4, MELS)
    assert lengths.tolist() == [4, 2] and labels.tolist() == [2, 3]
    assert not audio[1, 2:].any()
    mean, scale = fit_acoustic_normalizer([np.ones((2, MELS)), np.full((5, MELS), 12)], [0])
    assert np.all(mean == 1) and np.all(scale > 0)


def test_acoustic_padding_cannot_change_embedding():
    torch.manual_seed(8)
    model = SpeechModel().eval()
    frames = torch.randn(1, 7, MELS)
    x = torch.nn.functional.pad(frames, (0, 0, 0, 11), value=100)
    length = torch.tensor([7])
    torch.testing.assert_close(model.temporal_features(frames, length), model.temporal_features(x, length))
    assert model(x, length).shape == (1, len(EMOTIONS))


def test_frozen_speech_and_text_masked_pooling():
    values = torch.tensor([[[2., 4.], [6., 8.], [999., 999.]]])
    torch.testing.assert_close(masked_mean(values, torch.tensor([[1, 1, 0]])), torch.tensor([[4., 6.]]))
    model = SpeechModel()
    inputs = [np.zeros((6, MELS), dtype=np.float32)]
    data = AcousticData(inputs, np.array([0]), [0], (np.zeros(MELS, dtype=np.float32), np.ones(MELS, dtype=np.float32)))
    assert embeddings(model, data, torch.device("cpu"), 8).shape == (1, 128)
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_fusion_shapes_and_capacity():
    speech, text = torch.randn(2, 128), torch.randn(2, 384)
    fusion, control = FusionModel(), SpeechControl()
    assert fusion(speech, text).shape == TextModel()(text).shape == control(speech).shape == (2, 7)
    assert fusion.fused_features(speech, text).shape == (2, 64)
    parameters = lambda net: sum(p.numel() for p in net.parameters())
    assert abs(parameters(fusion) - parameters(control)) / parameters(fusion) < .01


def test_metrics_reconstruct_from_probabilities():
    rows = [{"id": str(i), "predicted": e if i != 0 else EMOTIONS[1], "true": e,
             **{f"p_{x}": float(x == (e if i else EMOTIONS[1])) for x in EMOTIONS}}
            for i, e in enumerate(EMOTIONS)]
    analysis.verified_predictions(rows, {r["id"]: {"emotion": r["true"]} for r in rows}, {r["id"] for r in rows})
    values, by_class, confusion = analysis.classification_metrics(rows)
    assert values["accuracy"] == values["balanced_accuracy"] == pytest.approx(6 / 7)
    assert by_class[0]["recall"] == 0 and confusion.sum() == len(EMOTIONS)
