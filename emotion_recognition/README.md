# Multimodal emotion recognition

A CPU study of acoustic cues and linguistic content on the Toronto Emotional Speech Set (TESS). The spoken carrier phrase is the same across the seven emotions for each target word. The study asks whether a frozen language representation still helps a learned speech classifier.

## Reproduce

From the repository root on Linux with Python 3.12:

```bash
python -m venv .venv-emotion
source .venv-emotion/bin/activate
python -m pip install -r emotion_recognition/requirements.txt
python emotion_recognition/run.py
python -m pytest -q emotion_recognition/tests
```

Preparation downloads the [Kaggle TESS archive](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) and the pinned MiniLM checkpoint. Pass `--archive /path/to/tess.zip` to use a local copy. Raw recordings, features and checkpoints stay under the ignored `work/` directory unless an external archive is supplied. Set `--device cuda` on an environment with CUDA-enabled PyTorch.

Continue from a checkpoint with `--stage train`. To isolate one run, use `--stage train --protocol word --seed 42 --model speech`. Fusion and the capacity control load the corresponding completed speech checkpoint. Run `--stage analyze` after all training groups finish to rebuild the tables and plots from recorded predictions. The default run prepares, trains and analyzes all groups.

## Dataset and protocol

The two archive trees contain 5,600 WAV entries. Their file and PCM hashes match pairwise, leaving 2,800 recordings from two speakers, 200 words and seven balanced emotions. Every selected WAV also matches the filename and MD5 in the original Borealis manifest. One filename, `OA_bite_neutral.wav`, lacks the final character of the speaker prefix. Its containing directory identifies it as OAF in both archive trees. The [original dataset description](https://doi.org/10.5683/SP2/E8H2MF) supplies the carrier phrase used to reconstruct transcripts. No separate transcripts are in the archive. An unconstrained PocketSphinx spot check on 12 clips from both speakers recovered the exact carrier phrase in seven clips and the complete sentence in five. The dataset description defines the reconstruction used in the models.

The primary train/validation/test counts are 1,960/420/420 with disjoint target words. For each speaker holdout the counts are 1,190/210/1,400. Validation stays within the training speaker. Both holdout tests include vocabulary seen in training. All three splits use optimizer seeds 42, 123 and 456. Checkpoints use validation macro F1 and then validation loss. A 512-clip CPU benchmark compared batch sizes 48, 96 and 128 to verify throughput before full training.

## Saved results

Accuracy and macro F1 are percentages. Each entry is the mean ± sample standard deviation over three fixed seeds.

| Split | Speech accuracy | Text accuracy | Fusion accuracy |
|---|---:|---:|---:|
| Word-disjoint | 99.84 ± 0.14 | 14.29 ± 0.00 | 99.76 ± 0.24 |
| Hold out OAF | 42.26 ± 6.04 | 14.29 ± 0.00 | 40.48 ± 7.84 |
| Hold out YAF | 37.10 ± 2.03 | 14.29 ± 0.00 | 36.00 ± 2.37 |

Word-disjoint macro F1 was 99.84 ± 0.14 for speech, 11.85 ± 1.06 for text and 99.76 ± 0.24 for fusion. See the [per-seed scores](Results/accuracy_tables.csv), [means and standard deviations](Results/aggregate_metrics.csv) and [per-emotion scores](Results/per_class_metrics.csv). In the primary split, anger had the lowest F1 in both speech and fusion. Happiness, neutral and surprise reached 100% F1 in both. Under speaker holdout, happiness had the lowest F1 for both models in both directions. The strongest class switched from neutral when testing OAF to disgust when testing YAF.

Each transcript has 14 recordings, two per emotion. Both label entropy and conditional entropy are about 2.807 bits. Empirical mutual information is zero, and the deterministic text-only bound, majority accuracy and uniform chance are all 14.29%. The trained text head reached that bound in every test. The [dataset audit](Results/dataset_audit.json) includes the conditional frequencies and the archive hash.

The capacity-matched speech control scored 99.84% on the primary split. Fusion minus speech was −0.08 percentage points, with a target-word bootstrap 95% interval of [−0.24, 0.00]. Fusion also trailed speech-only on both speaker holdouts. It was slightly below the capacity control when holding out OAF and slightly above it when holding out YAF. Masking speech brought fusion accuracy to 14.29% in every protocol. Masking text changed the means by at most 0.91 points. Five text permutations per seed kept the primary mean between 99.76% and 99.84%. Zero masks are a sensitivity check because they can create inputs outside the training distribution. [Ablations](Results/ablation_results.csv) · [Paired intervals](Results/paired_accuracy.csv)

The primary speech result was checked against held-out IDs, raw and decoded audio hashes, cached feature hashes, validation checkpoint selection and probability-based metric reconstruction. The training-label shuffle dropped speech accuracy to 21.90% in one run. That single randomized run is a negative control for the original labels. It does not estimate a precise null distribution for correlated utterances.

PCA and t-SNE use the same 420 held-out examples at seed 42. The text representation has only 30 distinct vectors, each shared by all seven emotions. The projection fits on distinct text vectors and draws coincident samples together. High-dimensional emotion silhouette was 0.653 for speech, −0.017 for text and 0.812 for fusion. The tighter fusion clusters did not translate to higher held-out accuracy. [Representations](Results/representations/separability.csv) · [Projection figure](Results/plots/representations.pdf)

The five failure cases prioritize distinct speech/fusion error patterns at seed 42. Each pattern uses a primary example when available, then a speaker-holdout case, ranked by wrong-class confidence. Both acoustic models labelled the word-disjoint `YAF_bean_anger` as disgust. Fusion recovered `YAF_hash_disgust` in the YAF holdout but changed `OAF_tape_neutral` from the correct speech prediction to disgust in the OAF holdout. The waveforms and RMS values describe the recordings without identifying the model's decision cues. [Failure cases](Results/failure_cases.csv) · [Acoustic views](Results/plots/failure_cases.pdf)

## Model sizes and compute

| Model | Trainable parameters | Frozen encoder parameters |
|---|---:|---:|
| Log-Mel convolution + BiGRU | 61,751 | 0 |
| MiniLM + text MLP | 18,823 | 22,713,216 |
| Feature fusion | 24,151 | 22,766,256 |
| Capacity-matched speech control | 24,215 | 53,040 |

Training covered 37 runs, including the label shuffle. The summed training and validation time was 1,079 s across worker processes. Peak process RSS was 995 MiB. Two single-threaded workers ran groups concurrently. The [run table](Results/accuracy_tables.csv) records selected epochs and timing scope. Frozen embeddings are cached, so text and fusion inference timings there measure the corresponding heads.

TESS contains two actresses, scripted speech and balanced emotion labels. The speaker holdouts test transfer between those two people. They cannot establish broader speaker or population generalization. Raw audio is excluded from Git. The [original release](https://doi.org/10.5683/SP2/E8H2MF) lists CC BY-NC 4.0, which should be checked before reusing or redistributing audio-derived artifacts.
