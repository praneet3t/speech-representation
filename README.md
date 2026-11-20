# Speech Representation

This project uses Montreal Forced Aligner (MFA) to put word and phone timestamps on six recordings from the supplied zip. There are 97.16 seconds of audio, 243 words and 150 distinct spellings.

## Run it

Extract the zip under `forced_alignment/data/` so its `wav/` and `transcripts/` folders sit together. Raw audio and transcripts stay local. The three NIST SPHERE files are copied to PCM WAV at their original 16 kHz.

```bash
conda env create -f forced_alignment/environment.yml
conda activate speech-align
python forced_alignment/run.py
```

Use `--wav PATH --transcripts PATH` to point to another extraction. The runner downloads the five pinned [model and dictionary assets](forced_alignment/scripts/dictionary.py) and checks their SHA-256 hashes. To run separately from the saved figures and TextGrids, append `--results forced_alignment/work/reproduction`. Run the tests with `python -m pytest -q` from the repo root.

## Results

All four runs aligned the same 243 word tokens. The baseline dictionary missed six spellings across seven tokens (2.9%). G2P filled those gaps in the corpus dictionary, and all seven `spn` intervals disappeared. This measures pronunciation coverage. Timing accuracy would require independent reference annotations.

US ARPA kept both boundaries unchanged for 64 of 243 words. Widening the baseline beams from 10/40 to 100/400 produced six byte-identical TextGrids. The largest shift is the end of “bed”. Baseline labels the final /d/ through 4.50 s, while US ARPA ends it at 2.06 s. Praat's waveform and spectrogram show a markedly quieter tail after 2.06 s. See the [boundary comparison](forced_alignment/Results/plots/boundary_examples.pdf), [saved metrics](forced_alignment/Results/summary.json) and [architecture notes](ARCHITECTURE.md).

![Baseline word and phone alignment for the “bed” clip](forced_alignment/Results/plots/alignments/ISLE_SESS0131_BLOCKD02_03_sprt1.png)

## Inspect in Praat

If the prepared WAV is missing, run `python forced_alignment/run.py --stage prepare --results forced_alignment/work/praat-prepare` after extracting the source data. From the repo root, open these files in Praat using **Open → Read from file**.

- Audio file `forced_alignment/work/run/corpus/ISLE_SESS0131_BLOCKD02_03_sprt1/ISLE_SESS0131_BLOCKD02_03_sprt1.wav`
- Baseline grid `forced_alignment/Results/TextGrids/baseline/ISLE_SESS0131_BLOCKD02_03_sprt1/ISLE_SESS0131_BLOCKD02_03_sprt1.TextGrid`
- US ARPA grid `forced_alignment/Results/TextGrids/alternate_model/ISLE_SESS0131_BLOCKD02_03_sprt1/ISLE_SESS0131_BLOCKD02_03_sprt1.TextGrid`

Select the Sound and one TextGrid in the Objects window, choose **View & Edit**, and enable **Spectrum → Show spectrogram**. Compare the word and phone tiers between 1.47 and 4.50 s, then repeat with the other grid. The saved inspection used Praat through Parselmouth.

TextGrids, pronunciation tables and labelled plots contain transcript-derived material. Confirm the source terms before redistributing them outside this private repository.

[MFA documentation](https://montreal-forced-aligner.readthedocs.io/) · [Official model releases](https://github.com/MontrealCorpusTools/mfa-models/releases) · [McAuliffe et al., 2017](https://doi.org/10.21437/Interspeech.2017-1386)
