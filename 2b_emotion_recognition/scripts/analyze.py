import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, silhouette_score)

from models.dataset import EMOTIONS, PROTOCOLS, SEEDS, file_hash, read_csv, verify_split, write_csv, write_json
from models.features import HOP, SAMPLE_RATE, log_mel_features, prepare_waveform


MAIN = ("speech", "text", "fusion")
CONTROL = "control"
NUMERICAL_METRICS = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")


def classification_metrics(rows):
    true = [r["true"] for r in rows]
    pred = [r["predicted"] for r in rows]
    precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=EMOTIONS, zero_division=0)
    stats = {"samples": len(rows), "accuracy": accuracy_score(true, pred),
             "balanced_accuracy": balanced_accuracy_score(true, pred),
             "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
             "macro_f1": float(np.mean(f1)), "weighted_f1": float(np.average(f1, weights=support))}
    per_emotion = [dict(emotion=e, precision=p, recall=r, f1=f, support=int(n))
                   for e, p, r, f, n in zip(EMOTIONS, precision, recall, f1, support, strict=True)]
    matrix = confusion_matrix(true, pred, labels=EMOTIONS)
    return stats, per_emotion, matrix


def verified_predictions(rows, manifest, test_ids):
    ids = [r["id"] for r in rows]
    if len(ids) != len(test_ids) or set(ids) != test_ids:
        raise ValueError("Predictions do not cover precisely the held-out IDs")
    values = np.array([[float(r[f"p_{e}"]) for e in EMOTIONS] for r in rows])
    true = [manifest[i]["emotion"] for i in ids]
    if (not np.isfinite(values).all() or np.min(values) < 0 or np.max(values) > 1
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
            or true != [r["true"] for r in rows]
            or [EMOTIONS[i] for i in values.argmax(axis=1)] != [r["predicted"] for r in rows]):
        raise ValueError("Label or probability mismatch")


def grouped_bootstrap(groups, protocol, manifest, comparison, repeats=2000):
    paired = defaultdict(list)
    for seed in SEEDS:
        control = {r["id"]: r for r in groups[protocol, seed, comparison]}
        for r in groups[protocol, seed, "fusion"]:
            a, b = r, control[r["id"]]
            paired[manifest[r["id"]]["word"]].append((a["predicted"] == a["true"]) - (b["predicted"] == b["true"]))
    # A sampled word carries all of its recordings and optimization seeds.
    totals = np.array([sum(values) for values in paired.values()])
    counts = np.array([len(values) for values in paired.values()])
    choices = np.random.default_rng(2026).integers(0, len(paired), size=(repeats, len(paired)))
    estimates = totals[choices].sum(axis=1) / counts[choices].sum(axis=1)
    low, high = np.quantile(estimates, [.025, .975])
    return {"protocol": protocol, "comparison": f"fusion_minus_{comparison}", "word_groups": len(paired),
            "replicates": repeats, "delta_accuracy": float(totals.sum() / counts.sum()),
            "ci_low": float(low), "ci_high": float(high)}


def representation_projections(output):
    path = output / "representations" / "word_seed42.npz"
    data = np.load(path)
    rows, stats = [], []
    for model in MAIN:
        features = data[model]
        unique, inverse = np.unique(features, axis=0, return_inverse=True)
        # Collapse identical text vectors before fitting either projection.
        space = unique if model == "text" else features
        pca = PCA(n_components=2, svd_solver="full")
        linear = pca.fit_transform(space)
        nonlinear = TSNE(n_components=2, perplexity=25, max_iter=1000, init="pca",
                         random_state=42, learning_rate="auto").fit_transform(space)
        if model == "text":
            linear, nonlinear = linear[inverse], nonlinear[inverse]
        stats.append({"model": model, "dimensions": features.shape[1], "samples": len(features),
                      "unique_vectors": len(unique), "silhouette_emotion": float(silhouette_score(features, data["labels"])),
                      "pca_variance_fraction": float(sum(pca.explained_variance_ratio_))})
        for i, identifier in enumerate(data["ids"]):
            rows.append({"model": model, "id": identifier, "emotion": data["labels"][i],
                         "word": data["words"][i], "pca_x": linear[i, 0], "pca_y": linear[i, 1],
                         "tsne_x": nonlinear[i, 0], "tsne_y": nonlinear[i, 1]})
    write_csv(output / "representations" / "projection.csv", rows)
    write_csv(output / "representations" / "separability.csv", stats)
    return rows


def failures(groups, manifest, archive_path, output):
    seed = SEEDS[0]
    predictions = {protocol: {model: {r["id"]: r for r in groups[protocol, seed, model]} for model in MAIN}
                   for protocol in PROTOCOLS}
    categories = defaultdict(list)
    for protocol, variants in predictions.items():
        for key, speech in variants["speech"].items():
            fusion, text = variants["fusion"][key], variants["text"][key]
            correct_speech = speech["predicted"] == speech["true"]
            correct_fusion = fusion["predicted"] == fusion["true"]
            if not correct_speech or not correct_fusion:
                pattern = {(False, True): "speech wrong, fusion right", (True, False): "speech right, fusion wrong",
                           (False, False): "both wrong"}[correct_speech, correct_fusion]
                confidence = max(float(r[f"p_{r['predicted']}"]) for r in (speech, fusion)
                                 if r["predicted"] != r["true"])
                categories[pattern].append((confidence, key, protocol))
            elif text["predicted"] != text["true"]:
                categories["text wrong"].append((float(text[f"p_{text['predicted']}"]), key, protocol))

    def order(item):
        confidence, key, protocol = item
        return (protocol != "word", -confidence, key, protocol)

    selected, used = [], set()
    for kind in ("speech wrong, fusion right", "speech right, fusion wrong", "both wrong"):
        ordered = sorted(categories[kind], key=order)
        candidate = next((item for item in ordered if item[1] not in used), None)
        if candidate is not None:
            selected.append((kind, candidate))
            used.add(candidate[1])
    extras = [(kind, item) for kind, values in categories.items() for item in values]
    extras.sort(key=lambda pair: (pair[0] == "text wrong", *order(pair[1])))
    for kind, item in extras:
        if len(selected) == 5:
            break
        if item[1] not in used:
            selected.append((kind, item))
            used.add(item[1])
    rows, plots = [], []
    with ZipFile(archive_path) as archive:
        for pattern, (confidence, key, protocol) in selected:
            item = manifest[key]
            raw, rate = sf.read(io.BytesIO(archive.read(item["archive_path"])), dtype="float32")
            wave = prepare_waveform(raw, rate)
            # The log-Mel function trims its input itself. Match its axes to the retained waveform.
            mel = log_mel_features(raw, rate)
            energy = 20 * np.log10(max(float(np.sqrt(np.mean(wave ** 2))), 1e-9))
            peak = 20 * np.log10(max(float(np.max(np.abs(wave))), 1e-9))
            quarter = len(wave) // 4
            tail = 20 * np.log10(max(float(np.sqrt(np.mean(wave[-quarter:] ** 2))), 1e-9))
            row = {"id": key, "protocol": protocol, "seed": seed, "category": pattern, "true": item["emotion"],
                   "transcript": item["transcript"], "incorrect_confidence": confidence,
                   "original_duration_s": len(raw) / rate, "retained_duration_s": len(wave) / SAMPLE_RATE,
                   "rms_dbfs": energy, "peak_dbfs": peak, "last_quarter_rms_dbfs": tail,
                   "acoustic_observation": f"Retained clip RMS {energy:.1f} dBFS. Last quarter RMS {tail:.1f} dBFS.",
                   "interpretation": "This waveform shows energy and timing. It cannot identify which cues drove the classifier."}
            for model in MAIN:
                p = predictions[protocol][model][key]
                row[f"{model}_prediction"] = p["predicted"]
                row[f"{model}_confidence"] = p[f"p_{p['predicted']}"]
                for emotion in EMOTIONS:
                    row[f"{model}_p_{emotion}"] = p[f"p_{emotion}"]
            rows.append(row)
            plots.append((row, wave, mel))
    write_csv(output / "failure_cases.csv", rows)
    return plots


def analyze(work, output, archive):
    config = json.loads((output / "experiment_config.json").read_text())
    if file_hash(archive) != json.loads((output / "dataset_audit.json").read_text())["archive_sha256"]:
        raise ValueError("Failure-case audio archive differs from the audited TESS archive")
    manifest_rows = read_csv(output / "manifest.csv")
    manifest = {row["id"]: row for row in manifest_rows}
    partitions = {}
    split_summary = {}
    for protocol in PROTOCOLS:
        split = read_csv(output / "splits" / f"{protocol}.csv")
        assert [r["id"] for r in split] == [r["id"] for r in manifest_rows]
        verify_split(manifest_rows, [r["partition"] for r in split], protocol)
        partitions[protocol] = {r["id"] for r in split if r["partition"] == "test"}
        split_summary[protocol] = {}
        for name in ("train", "validation", "test"):
            items = [r for r in split if r["partition"] == name]
            split_summary[protocol][name] = {
                "records": len(items), "words": len({r["word"] for r in items}),
                "speakers": sorted({r["speaker"] for r in items}),
                "by_emotion": {emotion: sum(r["emotion"] == emotion for r in items) for emotion in EMOTIONS}}
        words = [{r["word"] for r in split if r["partition"] == part} for part in ("train", "test")]
        split_summary[protocol]["train_test_word_overlap"] = len(words[0] & words[1])
    write_json(output / "split_summary.json", split_summary)
    audit = json.loads((output / "dataset_audit.json").read_text())
    groups, histories, resources = {}, [], {}
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            models = [*MAIN, CONTROL]
            if protocol == "word" and seed == SEEDS[0]:
                models.append("label_shuffle")
            for model in models:
                directory = work / "runs" / f"{protocol}-{seed}-{model}"
                runtime = json.loads((directory / "metrics.json").read_text())
                histories.extend(dict(protocol=protocol, seed=seed, model=model, **r)
                                 for r in read_csv(directory / "history.csv"))
                resources[protocol, seed, model] = runtime
                rows_by_model = defaultdict(list)
                for row in read_csv(directory / "predictions.csv"):
                    if row["protocol"] != protocol or int(row["seed"]) != seed:
                        raise ValueError("Wrong run in predictions")
                    rows_by_model[row["model"]].append(row)
                expected = {model}
                if model == "fusion":
                    expected |= {"mask_text", "mask_speech", *(f"permute_text_{i}" for i in config["permutation_seeds"])}
                if expected != rows_by_model.keys():
                    raise ValueError(f"Missing variant for {directory}")
                for variant, rows in rows_by_model.items():
                    verified_predictions(rows, manifest, partitions[protocol])
                    groups[protocol, seed, variant] = rows
    all_metrics, per_class, confusion = [], [], []
    for (protocol, seed, model), rows in groups.items():
        metrics, emotions, matrix = classification_metrics(rows)
        if model == "text":
            scores = {}
            for row in rows:
                phrase = manifest[row["id"]]["transcript"]
                value = np.array([float(row[f"p_{e}"]) for e in EMOTIONS])
                if not np.allclose(value, scores.setdefault(phrase, value), atol=1e-7, rtol=1e-6):
                    raise ValueError(f"Text-only probabilities vary within a transcript: {phrase}")
            if abs(metrics["accuracy"] - audit["deterministic_text_accuracy_bound"]) > 1e-9:
                raise ValueError("Text-only predictions disagree with the balanced transcript bound.")
        keys = {"protocol": protocol, "seed": seed, "model": model}
        all_metrics.append({**keys, **metrics, **resources.get((protocol, seed, model), {})})
        per_class.extend({**keys, **row} for row in emotions)
        confusion.extend({**keys, "true": true, "predicted": pred, "count": int(matrix[i, j])}
                         for i, true in enumerate(EMOTIONS) for j, pred in enumerate(EMOTIONS))
    write_csv(output / "predictions.csv", [r for rows in groups.values() for r in rows])
    write_csv(output / "training_history.csv", histories)
    write_csv(output / "accuracy_tables.csv", all_metrics)
    write_csv(output / "speaker_results.csv", [r for r in all_metrics if r["protocol"] != "word"])
    write_csv(output / "per_class_metrics.csv", per_class)
    write_csv(output / "confusions.csv", confusion)
    write_csv(output / "ablation_results.csv", [r for r in all_metrics if r["model"] not in {"text", "speech"}])
    write_csv(output / "paired_accuracy.csv", [grouped_bootstrap(groups, p, manifest, model)
                                                for p in PROTOCOLS for model in ("speech", "control")])
    combined = []
    for p in PROTOCOLS:
        variants = list(dict.fromkeys(r["model"] for r in all_metrics if r["protocol"] == p))
        for model in variants:
            rows = [r for r in all_metrics if r["protocol"] == p and r["model"] == model]
            combined.append({"protocol": p, "model": model, "seed_count": len(rows),
                             **{f"{m}_{stat}": float(func([r[m] for r in rows])) for m in NUMERICAL_METRICS
                                for stat, func in (("mean", np.mean), ("sd", lambda x: np.std(x, ddof=1) if len(x)>1 else 0))}})
    write_csv(output / "aggregate_metrics.csv", combined)
    projections = representation_projections(output)
    failure_plots = failures(groups, manifest, archive, output)
    from scripts.visualize import figures
    figures(output, audit, combined, all_metrics, per_class, histories, confusion, projections, failure_plots)
    print(f"Verified {len(groups)} held-out prediction blocks and produced tables and figures.", flush=True)
