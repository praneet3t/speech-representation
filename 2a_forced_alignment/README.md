# 2a · Forced alignment

Montreal Forced Aligner aligned 243 words across six recordings. Dictionary augmentation removed seven unknown-phone intervals. Wider beams produced identical TextGrids. These comparisons measure coverage and agreement. Timing accuracy requires reference annotations.

[Report](report.pdf) · [Results](Results/summary.json) · [TextGrids](Results/TextGrids) · [Figures](Results/plots)

## Run

Extract the source ZIP under `2a_forced_alignment/data/` with the `wav/` and `transcripts/` folders together, or pass their paths with `--wav` and `--transcripts`.

```bash
conda env create -f 2a_forced_alignment/environment.yml
conda activate speech-align
python 2a_forced_alignment/run.py --results 2a_forced_alignment/work/reproduction
python -m pytest -q 2a_forced_alignment/tests
```

## Praat

Open a prepared WAV from `work/run/corpus/` and its matching [TextGrid](Results/TextGrids). Select both in Praat and choose **View & Edit**. For the largest discrepancy, compare baseline and US ARPA on `ISLE_SESS0131_BLOCKD02_03_sprt1` between 1.47 and 4.50 s. The report's acoustic inspection used Parselmouth.

Check source terms before redistributing transcript-derived outputs.
