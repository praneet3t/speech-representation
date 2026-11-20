# 2b · Emotion recognition

Speech, text, and fusion models trained on 2,800 unique TESS recordings. Each transcript occurs equally often in all seven emotions.

| Test split | Speech | Text | Fusion |
|---|---:|---:|---:|
| Held-out words | 99.84 ± 0.14% | 14.29 ± 0.00% | 99.76 ± 0.24% |
| Hold out OAF | 42.26 ± 6.04% | 14.29 ± 0.00% | 40.48 ± 7.84% |
| Hold out YAF | 37.10 ± 2.03% | 14.29 ± 0.00% | 36.00 ± 2.37% |

Accuracy is mean ± sample standard deviation across three seeds per split. Text reached its 14.29% transcript-based ceiling. Fusion showed no consistent gain. The speaker results describe transfer between two actors.

[Report](report.pdf) · [Per-seed scores](Results/accuracy_tables.csv) · [Figures](Results/plots)

## Run

Use Python 3.12 in a separate environment. Preparation fetches [TESS](https://doi.org/10.5683/SP2/E8H2MF) and the pinned MiniLM checkpoint. Pass `--archive /path/to/tess.zip` to use a local copy.

```bash
python -m venv .venv-emotion
source .venv-emotion/bin/activate
python -m pip install -r 2b_emotion_recognition/requirements.txt
python 2b_emotion_recognition/run.py
python -m pytest -q 2b_emotion_recognition/tests
```

Use `--stage train` to resume checkpoints or `--stage analyze` to regenerate tables and figures. For a single run, add `--stage train --protocol word --seed 42 --model speech`. Raw audio and checkpoints stay outside Git. Check the TESS terms before redistributing audio-derived figures.
