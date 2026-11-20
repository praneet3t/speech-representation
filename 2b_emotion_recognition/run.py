import argparse
import hashlib
import json
import os
from pathlib import Path

# Limit numerical-library thread pools before importing the training stack.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from models import features
from models.dataset import EMOTIONS, PROTOCOLS, SEEDS, SPLIT_SEED, file_hash, read_csv, verify_split, write_csv, write_json
from models.fusion_pipeline import test as fusion_test, train as fusion_train
from models.speech_pipeline import test as speech_test, train as speech_train
from models.speech_pipeline.model import SpeechModel
from models.text_pipeline import test as text_test, train as text_train
from models.training import AcousticData, fit_acoustic_normalizer, fixed_data
from scripts.prepare_data import audit_archive, download_archive, extract_features


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
RESULTS = ROOT / "Results"
PERMUTATIONS = (11, 23, 37, 53, 79)
EPOCHS = 35
PATIENCE = 8
BATCH = 96


def prepare(archive, device):
    download_archive(archive)
    rows = audit_archive(archive, RESULTS, 2)
    extract_features(archive, rows, WORK, 2)
    meta = features.cache_text([r["transcript"] for r in rows], WORK, device)
    config = {
        "dataset_manifest_sha256": file_hash(RESULTS / "manifest.csv"),
        "dependencies": {"python": os.sys.version.split()[0], "torch": str(torch.__version__)},
        "acoustic": {"sample_rate": features.SAMPLE_RATE, "window_samples": features.WINDOW,
                     "hop_samples": features.HOP, "fft": features.FFT, "mel_bands": features.MELS,
                     "mel_range_hz": [30, 7800], "projection_stride": 3, "gru_hidden_per_direction": 64,
                     "silence_margin_ms": 100, "max_trim_per_edge_ms": 250,
                     "rms_activity_threshold": "max(0.001, peak_block_rms * 0.025)"},
        "text": meta, "seed": list(SEEDS), "split_seed": SPLIT_SEED, "permutation_seeds": list(PERMUTATIONS),
        "batch_size": BATCH, "max_epochs": EPOCHS, "patience": PATIENCE, "optimizer": "AdamW",
        "learning_rate": .001, "weight_decay": .01, "dropout": .25,
        "checkpoint_order": ["validation macro F1", "lower validation loss"],
        "label_shuffle": {"protocol": "word", "seed": SEEDS[0], "label_seed": 1729},
        "max_preprocess_threads": 2, "representation_seed": SEEDS[0],
        "representation_tsne": {"seed": 42, "perplexity": 25, "max_iter": 1000},
    }
    write_json(RESULTS / "experiment_config.json", config)
    write_json(WORK / "cache_hashes.json",
               {path: file_hash(WORK / path) for path in ("acoustic.npz", "contextual.npz")})
    print(f"Prepared {len(rows)} unique recordings and {meta['unique_transcripts']} contextual embeddings.", flush=True)


def prediction_rows(rows, indices, values, protocol, seed, name):
    return [{"id": rows[i]["id"], "protocol": protocol, "seed": seed, "model": name,
             "true": rows[i]["emotion"], "predicted": EMOTIONS[int(np.argmax(probability))],
             **{f"p_{label}": float(p) for label, p in zip(EMOTIONS, probability, strict=True)}}
            for i, probability in zip(indices, values, strict=True)]


def trained_encoder(directory, device, expected_settings):
    settings = json.loads((directory / "settings.json").read_text())
    expected = {**expected_settings, "device": str(device), "frozen_parameters": 0}
    if settings != expected:
        raise ValueError(f"Speech checkpoint settings differ: {directory}")
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    if not checkpoint["completed"]:
        raise ValueError(f"Finish speech training first: {directory}")
    encoder = SpeechModel()
    encoder.load_state_dict(checkpoint["best"]["model"])
    return encoder.to(device).eval().requires_grad_(False)


def train_group(protocol, seed, variant, device):
    rows = read_csv(RESULTS / "manifest.csv")
    config = json.loads((RESULTS / "experiment_config.json").read_text())
    cache = json.loads((WORK / "cache_hashes.json").read_text())
    if file_hash(RESULTS / "manifest.csv") != config["dataset_manifest_sha256"]:
        raise ValueError("Manifest changed. Prepare the corpus again.")
    for path, digest in cache.items():
        if file_hash(WORK / path) != digest:
            raise ValueError(f"Feature cache changed: {path}")
    indices_path = RESULTS / "splits" / f"{protocol}.csv"
    entries = read_csv(indices_path)
    if any(r["id"] != s["id"] for r, s in zip(rows, entries, strict=True)):
        raise ValueError("Partition order does not match the data")
    parts = np.array([row["partition"] for row in entries])
    verify_split(rows, parts, protocol)
    train, valid, test = [np.flatnonzero(parts == label) for label in ("train", "validation", "test")]
    labels = np.array([EMOTIONS.index(r["emotion"]) for r in rows])
    acoustic = np.load(WORK / "acoustic.npz")
    if acoustic["ids"].tolist() != [r["id"] for r in rows]:
        raise ValueError("Acoustic cache order does not match the manifest")
    offsets, frames = acoustic["offsets"], acoustic["frames"]
    speech_features = [frames[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    normalization = fit_acoustic_normalizer(speech_features, train)
    text_cache = np.load(WORK / "contextual.npz")
    text_map = dict(zip(text_cache["text"], text_cache["pooled"], strict=True))
    text = np.stack([text_map[r["transcript"]] for r in rows])
    text_parameters = config["text"]["frozen_parameters"]
    base = {"protocol": protocol, "seed": seed, "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE,
            "train_size": len(train), "validation_size": len(valid),
            "split_sha256": file_hash(indices_path), "feature_sha256": cache,
            "definition_sha256": hashlib.sha256(b"".join(
                (ROOT / part).read_bytes() for part in
                ("models/speech_pipeline/model.py", "models/text_pipeline/model.py",
                 "models/fusion_pipeline/model.py", "models/training.py"))).hexdigest()}
    runs = WORK / "runs"
    runs.mkdir(exist_ok=True)
    variants = ["speech", "text", "fusion", "control"] if variant == "all" else [variant]
    if variant == "all" and protocol == "word" and seed == SEEDS[0]:
        variants.append("label_shuffle")
    speech_model = None
    vector = None
    for name in variants:
        directory = runs / f"{protocol}-{seed}-{name}"
        settings = {**base, "model": name}
        if name in ("speech", "label_shuffle"):
            training_labels = labels.copy()
            if name == "label_shuffle":
                training_labels[train] = np.random.default_rng(1729).permutation(labels[train])
            training_data = AcousticData(speech_features, training_labels, train, normalization)
            validation = AcousticData(speech_features, labels, valid, normalization)
            model, history, info = speech_train.train(training_data, validation, directory, settings, device)
            held_out = AcousticData(speech_features, labels, test, normalization)
            values, seconds = speech_test.evaluate(model, held_out, device, BATCH)
            if name == "speech":
                speech_model = model
        else:
            if name == "text":
                arrays = (text,)
                model, history, info = text_train.train(fixed_data(arrays, labels, train),
                                                       fixed_data(arrays, labels, valid),
                                                       directory, settings, device, text_parameters)
                values, seconds = text_test.evaluate(model, fixed_data(arrays, labels, test), device, BATCH)
            else:
                if speech_model is None:
                    speech_model = trained_encoder(runs / f"{protocol}-{seed}-speech", device,
                                                   {**base, "model": "speech"})
                if vector is None:
                    acoustic_data = AcousticData(speech_features, labels, np.arange(len(rows)), normalization)
                    vector = speech_test.embeddings(speech_model, acoustic_data, device, BATCH)
                control = name == "control"
                arrays = (vector,) if control else (vector, text)
                frozen_speech = sum(p.numel() for layer in (speech_model.projection, speech_model.recurrent)
                                    for p in layer.parameters())
                frozen = frozen_speech + (0 if control else text_parameters)
                speech_checkpoint = runs / f"{protocol}-{seed}-speech" / "last.pt"
                settings["speech_checkpoint_sha256"] = file_hash(speech_checkpoint)
                model, history, info = fusion_train.train(fixed_data(arrays, labels, train),
                                                         fixed_data(arrays, labels, valid),
                                                         directory, settings, device, frozen, control)
                values, seconds = fusion_test.evaluate(model, fixed_data(arrays, labels, test), device, BATCH)
        preds = prediction_rows(rows, test, values, protocol, seed, name)
        if name == "fusion":
            sx, tx = vector[test], text[test]
            ablations = {"mask_text": (sx, np.zeros_like(tx)), "mask_speech": (np.zeros_like(sx), tx)}
            for perm_seed in PERMUTATIONS:
                order = np.random.default_rng(perm_seed).permutation(len(test))
                ablations[f"permute_text_{perm_seed}"] = (sx, tx[order])
            all_test = np.arange(len(test))
            for ablation, pair in ablations.items():
                predicted, _ = fusion_test.evaluate(model, fixed_data(pair, labels[test], all_test), device, BATCH)
                preds.extend(prediction_rows(rows, test, predicted, protocol, seed, ablation))
            if seed == SEEDS[0] and protocol == "word":
                dest = RESULTS / "representations"
                dest.mkdir(exist_ok=True)
                fused = fusion_test.representations(model, torch.from_numpy(sx), torch.from_numpy(tx), device, BATCH)
                np.savez_compressed(dest / "word_seed42.npz", speech=sx, text=tx, fusion=fused,
                                    labels=np.array([r["true"] for r in preds[:len(test)]]),
                                    ids=np.array([rows[i]["id"] for i in test]),
                                    words=np.array([rows[i]["word"] for i in test]))
        write_csv(directory / "history.csv", history)
        write_json(directory / "metrics.json", {**info, "model": name, "protocol": protocol, "seed": seed,
                                                "inference_seconds": seconds, "torch_threads": torch.get_num_threads(),
                                                "inference_scope": "acoustic GRU or cached-embedding head"})
        write_csv(directory / "predictions.csv", preds)


def main():
    parser = argparse.ArgumentParser(description=__doc__ or "Run the TESS emotion study.")
    parser.add_argument("--stage", choices=("all", "prepare", "train", "analyze"), default="all")
    parser.add_argument("--archive", type=Path, default=WORK / "tess.zip")
    parser.add_argument("--protocol", choices=("all", *PROTOCOLS), default="all")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--model", choices=("all", "speech", "text", "fusion", "control", "label_shuffle"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    if args.stage == "all" and (args.protocol != "all" or args.seed or args.model != "all"):
        parser.error("Select --stage train for partial runs.")
    if args.model == "label_shuffle" and (args.protocol != "word" or args.seed != SEEDS[0]):
        parser.error("Use --protocol word --seed 42 for the label-permutation sanity check.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(args.device)
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    if args.stage in ("all", "prepare"):
        prepare(args.archive.resolve(), device)
    if args.stage in ("all", "train"):
        for protocol in PROTOCOLS if args.protocol == "all" else (args.protocol,):
            for seed in SEEDS if args.seed is None else (args.seed,):
                train_group(protocol, seed, args.model, device)
    if args.stage in ("all", "analyze"):
        from scripts.analyze import analyze
        analyze(WORK, RESULTS, args.archive.resolve())


if __name__ == "__main__":
    main()
