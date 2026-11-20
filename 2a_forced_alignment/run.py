import argparse
import csv
from collections import Counter
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from scripts.align import CONFIGURATIONS, mfa_runner, run_alignment, validation_oovs
from scripts.dictionary import (RELEASES, build_dictionary, coverage, download_models,
                                missing_words, model_metadata, read_dictionary, validate_phones)
from scripts.prepare_data import discover_inputs, prepare, sha256


ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def alignment_checksums(results):
    inputs = ["dataset_inventory.csv", "experiment_config.json", "corpus.dict", "runtime.json"]
    grids = sorted(p.relative_to(results).as_posix() for p in (results / "TextGrids").rglob("*.TextGrid"))
    return {name: sha256(results / name) for name in inputs + grids}


def execute(args, results, work):
    models = work / "models"
    scratch = work / "run"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    results.mkdir(parents=True, exist_ok=True)
    corpus = scratch / "corpus"
    wav, transcripts = ((args.wav.resolve(), args.transcripts.resolve()) if args.wav
                        else discover_inputs(ROOT / "data"))
    rows, tokens, issues = prepare(wav, transcripts, corpus)
    write_csv(results / "dataset_inventory.csv", rows)
    audit = {"source": "supplied WAV/transcript directories", "recordings": sum(bool(r["audio"]) for r in rows),
             "transcripts": sum(bool(r["transcript"]) for r in rows), "pairs": len(tokens),
             "speaker_handling": "One recording per MFA speaker group. Speaker identities are unavailable.",
             "duplicate_recordings": [r["utterance"] for r in rows if r.get("duplicate_of")],
             "issues": issues, "duration_s": sum(r.get("duration_s", 0) for r in rows),
             "word_tokens": sum(map(len, tokens.values())),
             "word_types": len({w for text in tokens.values() for w in text})}
    write_json(results / "dataset_audit.json", audit)
    if issues:
        raise ValueError(f"Corpus has invalid or unmatched files: {', '.join(issues)}")
    if args.stage == "prepare":
        return

    run_mfa = mfa_runner(scratch)
    run_mfa(["version"], scratch / "version.log")
    version = (scratch / "version.log").read_text(encoding="utf-8").strip()
    if version != "3.3.9":
        raise RuntimeError(f"Use MFA 3.3.9 from environment.yml (found {version}).")
    download_models(models)
    metadata = {k: model_metadata(models / k) for k in RELEASES if k.endswith(".zip")}
    official = read_dictionary(models / "english_us_mfa.dict")
    alternate = read_dictionary(models / "english_us_arpa.dict")
    validate_phones(official, metadata["english_mfa.zip"])
    validate_phones(alternate, metadata["english_us_arpa.zip"])
    g2p = metadata["english_us_mfa_g2p.zip"]
    if set(g2p["phones"]) - set(metadata["english_mfa.zip"]["phones"]):
        raise ValueError("G2P phone inventory incompatible with the primary acoustic model")
    custom, provenance = build_dictionary(tokens, official, models, models / "corpus.dict", scratch, run_mfa)
    validate_phones(custom, metadata["english_mfa.zip"])
    shutil.copyfile(models / "corpus.dict", results / "corpus.dict")
    write_csv(results / "pronunciation_provenance.csv", provenance)
    dictionaries = dict(zip(("baseline", "corpus_dictionary", "alternate_model"),
                            (official, custom, alternate)))
    write_csv(results / "dictionary_coverage.csv",
              [{"configuration": name, **coverage(tokens, dictionary)} for name, dictionary in dictionaries.items()])
    write_csv(results / "utterance_oov.csv", [
        {"configuration": config, "utterance": name, "words": len(words),
         "oov_tokens": len(missing), "missing": " ".join(missing)}
        for config, dictionary in dictionaries.items() for name, words in tokens.items()
        for missing in [missing_words(words, dictionary)]])
    if args.stage == "dictionary":
        return

    jobs = min(args.jobs, len(tokens))
    settings = {"mfa": version, "python": platform.python_version(),
                "jobs": jobs, "platform": platform.machine(),
                "versions": {p: importlib.metadata.version(p) for p in
                             ("numpy", "soundfile", "praatio", "praat-parselmouth", "matplotlib")},
                "models": {k: {"release": tag, "artifact": asset, "sha256": digest, "metadata": metadata.get(k)}
                           for k, (tag, asset, _, digest) in RELEASES.items()},
                "corpus_dictionary_sha256": sha256(models / "corpus.dict"),
                "configurations": {}}
    for name, (model, dictionary, beam, retry) in CONFIGURATIONS.items():
        settings["configurations"][name] = dict(acoustic_model=model, dictionary=dictionary,
                                                 beam=beam, retry_beam=retry)
    write_json(results / "experiment_config.json", settings)
    runtimes, validations = {}, []
    grid_dir = results / "TextGrids"
    (results / "alignment_checksums.json").unlink(missing_ok=True)
    if grid_dir.exists():
        shutil.rmtree(grid_dir)
    for name, config in CONFIGURATIONS.items():
        runtimes[name] = run_alignment(name, config, corpus, models, results, scratch, jobs, run_mfa)
        folder = scratch / f"{name}_validation"
        counts = validation_oovs(folder)
        dictionary = dictionaries["baseline" if name == "decoding_sensitivity" else name]
        expected = Counter(w for words in tokens.values() for w in missing_words(words, dictionary))
        files = sorted(p.name for p in folder.iterdir() if p.suffix == ".csv" and p.stat().st_size)
        if counts != expected:
            raise ValueError(f"{name}: MFA's OOV counts differ from the independent audit.")
        validations.append({"configuration": name, "oov_types": len(counts), "oov_tokens": counts.total(),
                            "oovs": dict(counts), "diagnostic_csvs": files})
    write_json(results / "validation.json", validations)
    write_json(results / "runtime.json", runtimes)
    write_json(results / "alignment_checksums.json", alignment_checksums(results))
    if args.stage == "align":
        return
    summarize(results, corpus, plots=args.stage == "all")


def summarize(results, corpus, plots):
    from scripts.analyze import analyze

    data = analyze(results, corpus, write_csv)
    if plots:
        from scripts.visualize import visualize
        visualize(results, corpus, data)
    if any(not row["valid"] for row in data["metrics"]):
        raise RuntimeError("Some alignments are missing or invalid. See alignment_metrics.csv.")
    print(f"Verified {len(data['metrics'])} TextGrids in {results}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the MFA forced-alignment study.")
    parser.add_argument("--wav", type=Path)
    parser.add_argument("--transcripts", type=Path)
    parser.add_argument("--jobs", type=int, default=min(3, os.cpu_count() or 1))
    parser.add_argument("--results", type=Path, default=ROOT / "Results")
    parser.add_argument("--stage", choices=["all", "prepare", "dictionary", "align", "analyze", "plot"], default="all")
    args = parser.parse_args()
    if (args.wav is None) != (args.transcripts is None) or args.jobs < 1:
        parser.error("Pass --wav and --transcripts together, and use at least one job.")
    results = args.results.resolve()
    work = ROOT / "work"
    if args.stage in {"analyze", "plot"}:
        manifest = results / "alignment_checksums.json"
        if not manifest.exists() or json.loads(manifest.read_text()) != alignment_checksums(results):
            raise RuntimeError("The recorded alignment run does not match these artifacts. Run --stage align first.")
        summarize(results, work / "run" / "corpus", plots=args.stage == "plot")
    else:
        execute(args, results, work)


if __name__ == "__main__":
    sys.exit(main())
