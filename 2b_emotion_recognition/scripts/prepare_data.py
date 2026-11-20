import hashlib
import io
import os
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from models.dataset import (EMOTIONS, PROTOCOLS, SPEAKERS, file_hash, linguistic_audit,
                            make_split, parse_filename, transcript, write_csv, write_json)
from models.features import log_mel_features

SOURCE = "https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess"


def download_archive(path):
    if not path.is_file():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(DOWNLOAD, timeout=90) as source, temporary.open("wb") as dest:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                dest.write(chunk)
        temporary.replace(path)


def inspect_recording(item):
    path, raw = item
    speaker, word, emotion = parse_filename(path)
    samples, rate = sf.read(io.BytesIO(raw), dtype="int32", always_2d=True)
    if not len(samples) or rate < 1:
        raise ValueError(f"Empty audio: {path}")
    pcm = hashlib.sha256(f"{rate}:{samples.shape[1]}:".encode() + samples.astype("<i4").tobytes()).hexdigest()
    row = {"id": f"{speaker}_{word}_{emotion}", "speaker": speaker, "word": word, "emotion": emotion,
           "transcript": transcript(word), "archive_path": path, "duration_s": len(samples) / rate,
           "sample_rate_hz": rate, "channels": samples.shape[1], "frames": len(samples),
           "sha256": hashlib.sha256(raw).hexdigest(), "pcm_sha256": pcm}
    return row


def select_unique(rows):
    by_id, by_pcm = {}, {}
    duplicates = 0
    for row in rows:
        key, digest = row["id"], row["pcm_sha256"]
        if key in by_id:
            if by_id[key]["pcm_sha256"] != digest:
                raise ValueError(f"Differing waveforms for {key}")
            duplicates += 1
        elif digest in by_pcm:
            raise ValueError(f"Shared waveform for different IDs: {key}, {by_pcm[digest]}")
        else:
            by_id[key], by_pcm[digest] = row, key
    return [by_id[k] for k in sorted(by_id)], duplicates


def audit_archive(archive, output, workers):
    with ZipFile(archive) as source, ThreadPoolExecutor(max_workers=workers) as workers_pool:
        files = sorted(n for n in source.namelist() if n.lower().endswith(".wav"))
        crc_error = source.testzip()
        if crc_error:
            raise ValueError(f"ZIP CRC failed: {crc_error}")
        rows = list(workers_pool.map(inspect_recording, ((n, source.read(n)) for n in files)))
        other_files = [n for n in source.namelist() if not n.endswith("/") and n not in files]
    selected, duplicates = select_unique(rows)
    words = sorted({r["word"] for r in selected})
    combinations = {(r["speaker"], r["word"], r["emotion"]) for r in selected}
    missing = [(s, w, e) for s in SPEAKERS for w in words for e in EMOTIONS if (s, w, e) not in combinations]
    if len(words) != 200 or missing:
        raise ValueError(f"Incomplete dataset: {len(words)} words, {len(missing)} missing triples")
    durations = [r["duration_s"] for r in selected]
    write_csv(output / "manifest.csv", selected)
    write_json(output / "dataset_audit.json", {
        "source": SOURCE, "carrier_phrase_documentation": "https://tspace.library.utoronto.ca/handle/1807/24487",
        "archive_sha256": file_hash(archive), "archive_wavs": len(rows), "recordings": len(selected),
        "duplicate_file_copies": len(rows) - len({r["sha256"] for r in rows}),
        "duplicate_decoded_copies": len(rows) - len({r["pcm_sha256"] for r in rows}),
        "duplicate_combinations": duplicates, "missing_combinations": missing, "corrupt_recordings": 0,
        "trees": dict(Counter(r["archive_path"].split("/")[0] for r in rows)),
        "sample_rates": dict(Counter(r["sample_rate_hz"] for r in selected)),
        "channels": dict(Counter(r["channels"] for r in selected)), "target_words": len(words),
        "duration_s": {"min": min(durations), "max": max(durations), "total": sum(durations),
                       "median": float(np.median(durations))},
        "by_emotion": dict(Counter(r["emotion"] for r in selected)),
        "by_speaker_emotion": {s: dict(Counter(r["emotion"] for r in selected if r["speaker"] == s))
                               for s in SPEAKERS},
        "standalone_transcripts": False, "other_archive_files": other_files,
        **linguistic_audit(selected),
    })
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    for protocol in PROTOCOLS:
        split = make_split(selected, protocol)
        write_csv(split_dir / f"{protocol}.csv",
                  [{**{k: r[k] for k in ("id", "word", "speaker", "emotion", "pcm_sha256")}, "partition": part}
                   for r, part in zip(selected, split)])
    return selected


def extract_features(archive, rows, work, workers):
    def feature(data):
        samples, rate = sf.read(io.BytesIO(data), dtype="float32")
        return log_mel_features(samples, rate)

    with ZipFile(archive) as source, ThreadPoolExecutor(max_workers=workers) as pool:
        features = list(pool.map(feature, (source.read(r["archive_path"]) for r in rows)))
    offsets = np.cumsum([0] + [len(f) for f in features])
    path = work / "acoustic.tmp.npz"
    np.savez(path, offsets=offsets, frames=np.concatenate(features),
             ids=np.array([r["id"] for r in rows]))
    os.replace(path, work / "acoustic.npz")
