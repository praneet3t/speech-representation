from collections import Counter
import os
import subprocess
import time


CONFIGURATIONS = {
    "baseline": ("english_mfa.zip", "english_us_mfa.dict", 10, 40),
    "corpus_dictionary": ("english_mfa.zip", "corpus.dict", 10, 40),
    "alternate_model": ("english_us_arpa.zip", "english_us_arpa.dict", 10, 40),
    "decoding_sensitivity": ("english_mfa.zip", "english_us_mfa.dict", 100, 400),
}


def mfa_runner(work):
    def run(args, log):
        env = os.environ | {"MFA_ROOT_DIR": str(work / "mfa"),
                            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        print(f"MFA {args[0]} → {log.name}", flush=True)
        started = time.perf_counter()
        with log.open("w", encoding="utf-8") as output:
            result = subprocess.run(["mfa", *args], stdout=output, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"MFA {args[0]} failed ({result.returncode}). See {log}.")
        return time.perf_counter() - started
    return run


def run_alignment(name, config, corpus, models, results, work, jobs, run_mfa):
    acoustic, dictionary, beam, retry_beam = config
    target = results / "TextGrids" / name
    target.mkdir(parents=True, exist_ok=True)
    validation = work / f"{name}_validation"
    model = str(models / acoustic)
    config_path = work / f"{name}.yaml"
    config_path.write_text(f"beam: {beam}\nretry_beam: {retry_beam}\n", encoding="utf-8")
    options = ["--clean", "--overwrite", "--num_jobs", str(jobs), "--config_path", str(config_path)]
    inputs = [str(corpus), str(models / dictionary)]
    validation_seconds = run_mfa(
        ["validate", *options, "--acoustic_model_path", model,
         "--output_directory", str(validation), *inputs], work / f"{name}_validate.log")
    seconds = run_mfa(
        ["align", *options, "--output_format", "long_textgrid", *inputs, model, str(target)],
        work / f"{name}_align.log")
    return {"validation_seconds": validation_seconds, "alignment_seconds": seconds}


def validation_oovs(folder):
    counts = Counter()
    for path in folder.glob("oov_counts_*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            word, count = line.split("\t")
            counts[word] += int(count)
    return counts
