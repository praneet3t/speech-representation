import csv
import hashlib
import json

import numpy as np
import soundfile as sf
from praatio import textgrid

from scripts.align import CONFIGURATIONS
from scripts.prepare_data import sha256


SILENCE = {"", "sil", "<sil>", "<eps>"}
UNKNOWN = {"<unk>", "<oov>", "spn"}
NON_SPEECH = {"[bracketed]", "[laughter]", "<noise>", "nsn"}


def read_textgrid(path):
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    tiers = {}
    for name in grid.tierNames:
        label = name.casefold().rsplit(" - ", 1)[-1].strip()
        key = {"word": "words", "word intervals": "words", "phone": "phones",
               "phonemes": "phones", "phoneme": "phones"}.get(label, label)
        if key in {"words", "phones"}:
            tier = grid.getTier(name)
            if key in tiers or not isinstance(tier, textgrid.IntervalTier):
                raise ValueError(f"Expected one interval tier for {key}")
            tiers[key] = list(tier.entries)
    if set(tiers) != {"words", "phones"}:
        raise ValueError(f"Missing word or phone tier in {path.name}")
    return grid.maxTimestamp, tiers


def timestamp_issues(entries, duration, tolerance=0.001):
    errors = []
    previous_end = None
    for i, (start, end, label) in enumerate(entries):
        if not np.isfinite([start, end]).all():
            errors.append(f"non-finite timestamp at {i}")
            continue
        if start < -tolerance or end > duration + tolerance:
            errors.append(f"out of bounds at {i}")
        if end < start or (label and end == start):
            errors.append(f"nonpositive interval at {i}")
        if previous_end is not None and start < previous_end - tolerance:
            errors.append(f"overlap or reversed order at {i}")
        previous_end = end if previous_end is None else max(previous_end, end)
    return errors


def word_intervals(entries):
    words = []
    for start, end, label in entries:
        label = label.casefold()
        if label in SILENCE | NON_SPEECH:
            continue
        if label == "'s" and words and abs(words[-1][1] - start) < 0.001:
            old_start, _, base = words[-1]
            words[-1] = old_start, end, base + label
        else:
            words.append((start, end, label))
    return words


def match_words(left, right):
    """Return ordered index pairs from a longest common subsequence."""
    scores = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for i in range(len(left) - 1, -1, -1):
        for j in range(len(right) - 1, -1, -1):
            scores[i][j] = (1 + scores[i + 1][j + 1] if left[i] == right[j]
                            else max(scores[i + 1][j], scores[i][j + 1]))
    pairs = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif scores[i + 1][j] >= scores[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def compare_boundaries(reference, comparison):
    matches = match_words([w[2] for w in reference], [w[2] for w in comparison])
    for i, j in matches:
        start, end, word = reference[i]
        new_start, new_end, _ = comparison[j]
        yield {"baseline_index": i + 1, "comparison_index": j + 1, "word": word,
               "baseline_start_s": start, "baseline_end_s": end,
               "comparison_start_s": new_start, "comparison_end_s": new_end,
               "start_abs_ms": round(1000 * abs(start - new_start), 6),
               "end_abs_ms": round(1000 * abs(end - new_end), 6),
               "duration_delta_ms": round(1000 * ((new_end - new_start) - (end - start)), 6)}


def silence_durations(entries, duration):
    speech = [(start, end) for start, end, label in entries if label.casefold() not in SILENCE]
    if not speech:
        return duration, 0.0, 0.0
    leading = max(0, speech[0][0])
    trailing = max(0, duration - speech[-1][1])
    internal = sum(max(0, right[0] - left[1]) for left, right in zip(speech, speech[1:]))
    return leading, internal, trailing


def distribution(values):
    if not len(values):
        return {"q1": None, "median": None, "q3": None}
    return {k: float(v) for k, v in zip(("q1", "median", "q3"), np.percentile(values, [25, 50, 75]))}


def inspect_grid(path, duration, expected, known_phones):
    grid_end, tiers = read_textgrid(path)
    words = word_intervals(tiers["words"])
    phones = [entry for entry in tiers["phones"] if entry[2].casefold() not in SILENCE | NON_SPEECH]
    problems = [f"{key}: {issue}" for key, entries in tiers.items()
                for issue in timestamp_issues(entries, duration)]
    if abs(grid_end - duration) > 0.001:
        problems.append("TextGrid duration differs from WAV")
    if not words or not phones:
        problems.append("Empty word or phone tier")
    pairs = match_words(expected, [w[2] for w in words])
    matched = len(pairs)
    leading, internal, trailing = silence_durations(tiers["phones"], duration)
    lexical = [w for w in words if w[2] not in UNKNOWN]
    speech_phones = [p for p in phones if p[2].casefold() not in UNKNOWN]
    unknown_words = len(words) - len(lexical)
    unknown_phones = [p for p in phones if p[2].casefold() in UNKNOWN]
    matched_intervals = [words[j] for _, j in pairs if words[j][2] not in UNKNOWN]
    words_with_unknown_phones = sum(any(min(end, pe) - max(start, ps) > .001
                                       for ps, pe, _ in unknown_phones)
                                   for start, end, _ in matched_intervals)
    other_phones = sorted({p[2] for p in speech_phones} - set(known_phones))
    metrics = {"valid": not problems, "issues": "; ".join(problems), "grid_duration_s": grid_end,
               "expected_words": len(expected), "word_intervals": len(words),
               "aligned_words": len(lexical), "aligned_phones": len(speech_phones),
               "matched_words": matched, "missing_transcript_words": len(expected) - matched,
               "unmatched_aligned_words": len(words) - matched,
               "word_alignment_coverage": len(matched_intervals) / len(expected),
               "transcript_agreement": 2 * matched / (len(expected) + len(words)), "unknown_words": unknown_words,
               "unknown_phones": len(unknown_phones), "words_with_unknown_phones": words_with_unknown_phones,
               "unexpected_phone_labels": " ".join(other_phones),
               "short_phones_lt_30ms": sum(round(e - s, 4) < .03 for s, e, _ in speech_phones),
               "long_phones_gt_300ms": sum(round(e - s, 4) > .3 for s, e, _ in speech_phones),
               "leading_silence_s": leading, "internal_silence_s": internal,
               "trailing_silence_s": trailing, "silence_fraction": (leading + internal + trailing) / duration}
    return metrics, words, speech_phones, tiers, unknown_phones


def analyze(results, corpus, write_csv):
    with (results / "dataset_inventory.csv").open(encoding="utf-8", newline="") as stream:
        inventory = list(csv.DictReader(stream))
    runtime = json.loads((results / "runtime.json").read_text())
    settings = json.loads((results / "experiment_config.json").read_text())
    for row in inventory:
        wav = corpus / row["utterance"] / f"{row['utterance']}.wav"
        # Match the working audio to the inventory before analyzing a previous run.
        pcm, rate = sf.read(wav, dtype="int16")
        lab = wav.with_suffix(".lab")
        if rate != int(row["sample_rate_hz"]) or len(pcm) != int(row["frames"]):
            raise ValueError(f"Prepared WAV timing has changed: {wav}")
        if row.get("normalized_sha256") and sha256(lab) != row["normalized_sha256"]:
            raise ValueError(f"Prepared transcript has changed: {lab}")
        if hashlib.sha256(np.asarray(pcm, dtype="<i2").tobytes()).hexdigest() != row["pcm_sha256"]:
            raise ValueError(f"Prepared WAV has changed: {wav}")
    metrics, grids, aggregates = [], {}, {}
    utterances = {r["utterance"]: float(r["duration_s"]) for r in inventory}
    for config, (acoustic, _, _, _) in CONFIGURATIONS.items():
        grids[config] = {}
        found = {}
        for path in (results / "TextGrids" / config).rglob("*.TextGrid"):
            if path.stem in found or path.stem not in utterances:
                raise ValueError(f"Unexpected or duplicate TextGrid: {path}")
            found[path.stem] = path
        config_metrics = []
        for name, duration in utterances.items():
            expected = (corpus / name / f"{name}.lab").read_text(encoding="utf-8").split()
            try:
                if name not in found:
                    raise ValueError("No TextGrid generated")
                values = inspect_grid(found[name], duration, expected, settings["models"][acoustic]["metadata"]["phones"])
                row = values[0] | {"textgrid_sha256": sha256(found[name])}
                if row["valid"]:
                    grids[config][name] = values[1:]
            except (ValueError, OSError, TypeError, IndexError, KeyError, textgrid.errors.PraatioException) as error:
                row = {"valid": False, "issues": str(error), "expected_words": len(expected)}
            config_metrics.append({"configuration": config, "utterance": name, "audio_duration_s": duration, **row})
        metrics.extend(config_metrics)
        valid = [r for r in config_metrics if r["valid"]]
        word_ms = [1000 * (end - start) for words, *_ in grids[config].values()
                   for start, end, word in words if word not in UNKNOWN]
        phone_ms = [1000 * (end - start) for _, phones, *_ in grids[config].values()
                    for start, end, _ in phones]
        elapsed = runtime[config]["alignment_seconds"]
        duration_total = sum(utterances.values())
        aggregates[config] = {
            **runtime[config], "real_time_factor": elapsed / duration_total,
            "utterances": len(utterances), "aligned_utterances": len(valid),
            "completion_rate": len(valid) / len(utterances),
            **{key: sum(r[key] for r in valid) for key in
               ("aligned_words", "word_intervals", "aligned_phones", "matched_words", "unknown_words",
                "unknown_phones", "words_with_unknown_phones")},
            "expected_words": sum(r["expected_words"] for r in config_metrics),
            "word_ms": distribution(word_ms), "phone_ms": distribution(phone_ms),
            "silence_fraction": sum(r["audio_duration_s"] * r["silence_fraction"] for r in valid)
            / sum(r["audio_duration_s"] for r in valid) if valid else None,
        }
    write_csv(results / "alignment_metrics.csv", metrics)
    for row in aggregates.values():
        row["transcript_agreement"] = 2 * row["matched_words"] / (row["expected_words"] + row["word_intervals"])
        row["word_alignment_coverage"] = row["matched_words"] / row["expected_words"]
        row["lexical_pronunciation_coverage"] = (
            (row["matched_words"] - row["words_with_unknown_phones"]) / row["expected_words"])

    comparisons, per_utterance, boundary_summary = [], [], {}
    for config in list(CONFIGURATIONS)[1:]:
        rows = []
        for name in utterances:
            left = grids["baseline"].get(name, ([],))[0]
            right = grids[config].get(name, ([],))[0]
            # A placeholder is evidence of an alignment, but its label does not identify a word.
            pairs = list(compare_boundaries(left, right))
            pairs = [r for r in pairs if r["word"] not in UNKNOWN]
            rows.extend({"configuration": config, "utterance": name, **r} for r in pairs)
            per_utterance.append({"configuration": config, "utterance": name, "matched_words": len(pairs),
                                  "unmatched_baseline": len(left) - len(pairs),
                                  "unmatched_comparison": len(right) - len(pairs),
                                  **{f"{metric}_{q}": value
                                     for metric in ("start_abs_ms", "end_abs_ms", "duration_delta_ms")
                                     for q, value in distribution([r[metric] for r in pairs]).items()},
                                  "max_boundary_ms": max((max(r["start_abs_ms"], r["end_abs_ms"])
                                                          for r in pairs), default=None)})
        comparisons.extend(rows)
        totals = [r for r in per_utterance if r["configuration"] == config]
        boundary_summary[config] = {
            "matched_words": len(rows), "unmatched_baseline": sum(r["unmatched_baseline"] for r in totals),
            "unmatched_comparison": sum(r["unmatched_comparison"] for r in totals),
            **{metric: distribution([r[metric] for r in rows])
               for metric in ("start_abs_ms", "end_abs_ms", "duration_delta_ms")},
            "unchanged_both_boundaries": sum(r["start_abs_ms"] == r["end_abs_ms"] == 0 for r in rows),
        }
    if comparisons:
        write_csv(results / "boundary_comparison.csv", comparisons)
    write_csv(results / "boundary_per_utterance.csv", per_utterance)
    (results / "summary.json").write_text(json.dumps({"alignments": aggregates, "boundaries": boundary_summary},
                                                     indent=2) + "\n", encoding="utf-8")
    observations = select_observations(comparisons, grids, corpus)
    if observations:
        write_csv(results / "observations.csv", observations)
    return {"metrics": metrics, "grids": grids, "aggregate": aggregates, "boundaries": comparisons,
            "inventory": inventory, "observations": observations}


def segment_rms_db(waveform, rate, start, end):
    samples = waveform[round(start * rate):round(end * rate)]
    return round(20 * np.log10(max(np.sqrt(np.mean(samples ** 2)), 1e-12)), 2)


def select_observations(comparisons, grids, corpus):
    alternate = [r for r in comparisons if r["configuration"] == "alternate_model"]
    def displacement(row):
        return max(row["start_abs_ms"], row["end_abs_ms"])

    observations = []
    stable = [r for r in alternate if displacement(r) == 0]
    if stable:
        row = stable[0]
        observations.append({"case": "stable", **row,
                             "observation": "Both configurations place both boundaries at the same times."})
    if alternate and max(map(displacement, alternate)) > 0:
        row = max(alternate, key=displacement)
        wav, rate = sf.read(corpus / row["utterance"] / f"{row['utterance']}.wav")
        early, late = sorted((row["baseline_end_s"], row["comparison_end_s"]))
        observation = "Largest absolute matched-word boundary shift between configurations."
        if late - early > .3:
            interval_rms = segment_rms_db(wav, rate, early, late)
            spoken_rms = segment_rms_db(wav, rate, min(row["baseline_start_s"], row["comparison_start_s"]), early)
            observation += (f" The segment between the two word ends has {interval_rms:.1f} dBFS RMS."
                            f" The preceding shared word region has {spoken_rms:.1f} dBFS RMS.")
        observations.append({"case": "largest shift", **row, "observation": observation})
    for name, (_, _, _, unknown) in grids["baseline"].items():
        if unknown:
            start, end, phone = unknown[0]
            words = grids["baseline"][name][0]
            labels = [word for a, b, word in words if min(end, b) - max(start, a) > .001]
            observations.append({"case": "unknown pronunciation", "configuration": "baseline",
                                 "utterance": name, "word": " ".join(labels) or phone, "baseline_start_s": start, "baseline_end_s": end,
                                 "observation": "An unknown-phone interval marks speech with no lexical pronunciation."})
            break
    return observations
