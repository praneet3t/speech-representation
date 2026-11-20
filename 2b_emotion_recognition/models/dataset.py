import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


EMOTIONS = ("anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise")
LABELS = dict(zip(("angry", "disgust", "fear", "happy", "neutral", "sad", "ps"), EMOTIONS))
SPEAKERS = ("OAF", "YAF")
PROTOCOLS = ("word", "holdout_oaf", "holdout_yaf")
SEEDS = (42, 123, 456)
SPLIT_SEED = 2026


def parse_filename(filename):
    path = Path(filename)
    name = path.name
    if name == "OA_bite_neutral.wav" and path.parent.name == "OAF_neutral":
        name = "OAF_bite_neutral.wav"
    match = re.fullmatch(r"(OAF|YAF)_([a-z]+)_(angry|disgust|fear|happy|neutral|sad|ps)\.wav",
                         name, flags=re.I)
    if not match:
        raise ValueError(f"Unexpected TESS filename: {filename}")
    speaker, word, emotion = match.groups()
    return speaker.upper(), word.lower(), LABELS[emotion.lower()]


def transcript(word):
    return f"Say the word {word}"


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def make_split(rows, protocol):
    words = np.array(sorted({row["word"] for row in rows}))
    words = np.random.default_rng(SPLIT_SEED).permutation(words)
    if protocol == "word":
        train_words = set(words[:round(.70 * len(words))])
        valid_words = set(words[round(.70 * len(words)):round(.85 * len(words))])
        split = ["train" if r["word"] in train_words else
                 "validation" if r["word"] in valid_words else "test" for r in rows]
    else:
        held_out = {"holdout_oaf": "OAF", "holdout_yaf": "YAF"}[protocol]
        valid_words = set(words[:round(.15 * len(words))])
        split = ["test" if r["speaker"] == held_out else
                 "validation" if r["word"] in valid_words else "train" for r in rows]
    verify_split(rows, split, protocol)
    return split


def verify_split(rows, split, protocol):
    fields = ["id", "pcm_sha256", "word"] if protocol == "word" else ["id", "pcm_sha256"]
    for field in fields:
        memberships = {}
        for row, part in zip(rows, split, strict=True):
            value = row[field]
            if memberships.setdefault(value, part) != part:
                raise ValueError(f"Leaked {field}: {value}")
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Empty partition")
    if protocol != "word":
        test = {r["speaker"] for r, part in zip(rows, split) if part == "test"}
        train = {r["speaker"] for r, part in zip(rows, split) if part != "test"}
        if test & train or test != {protocol.rsplit("_", 1)[1].upper()}:
            raise ValueError("Speaker holdout mismatch")


def linguistic_audit(rows):
    groups = {}
    total = Counter()
    for row in rows:
        groups.setdefault(row["transcript"], Counter())[row["emotion"]] += 1
        total[row["emotion"]] += 1

    def entropy(counts):
        probabilities = np.array(list(counts.values())) / counts.total()
        return float(-(probabilities * np.log2(probabilities)).sum())

    count = len(rows)
    label_entropy = entropy(total)
    conditional = sum(group.total() * entropy(group) for group in groups.values()) / count
    return {
        "unique_transcripts": len(groups),
        "recordings_per_transcript": dict(Counter(g.total() for g in groups.values())),
        "label_entropy_bits": label_entropy,
        "conditional_entropy_bits": conditional,
        "mutual_information_bits": max(0., label_entropy - conditional),
        "deterministic_text_accuracy_bound": sum(max(g.values()) for g in groups.values()) / count,
        "uniform_chance_accuracy": 1 / len(EMOTIONS),
        "majority_accuracy": max(total.values()) / count,
        "conditional_probabilities": {
            t: {e: g[e] / g.total() for e in EMOTIONS} for t, g in sorted(groups.items())
        },
    }
