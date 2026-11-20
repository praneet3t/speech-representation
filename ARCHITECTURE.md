# How forced alignment works

Forced alignment starts with a recording and a known transcript. It estimates when each word and phone starts and ends. Speech recognition also has to decide which words were spoken. Here the transcript fixes the word order, while the aligner chooses among possible pronunciations, silence placements and durations.

## Audio and pronunciations

[prepare_data.py](forced_alignment/scripts/prepare_data.py) pairs audio and transcripts by file stem, normalizes the text and writes one WAV and one `.lab` file per utterance. The three NIST SPHERE recordings are converted to PCM WAV at the original 16 kHz. The project checks that the samples are unchanged. The corpus has no verified speaker IDs, so each recording forms its own MFA speaker group.

MFA computes mel-frequency cepstral coefficients (MFCCs) on overlapping 25 ms windows with a 10 ms hop. It measures short-time spectral energy with mel-spaced filters, takes logarithms and applies a cosine transform to summarize the spectral envelope. Both model packages specify 13 coefficients, cepstral mean and variance normalization, and delta features. The hop sets the granularity of the frame sequence. The spectrogram in the plots is a separate Praat visualization of the waveform.

A pronunciation dictionary maps a word such as `bed` to one or more phone sequences, such as `b ɛ d` in the MFA phone set or `B EH1 D` in ARPA. [dictionary.py](forced_alignment/scripts/dictionary.py) measures vocabulary coverage, preserves the official entries for covered words and calls `mfa g2p` for out-of-vocabulary (OOV) spellings in the corpus-dictionary run. The pretrained English US MFA G2P 3.0.0 model is a Phonetisaurus weighted transducer that predicts phone sequences from spelling. These candidates give the decoder paths it can score against the audio. Names still need pronunciation review. Without a lexical pronunciation, MFA can keep the word label and use `spn` as its phone.

## Acoustic models and decoding

The baseline uses the English MFA acoustic model 3.1.0 and the English US MFA dictionary 3.1.0 with an IPA-based inventory. The model was trained on several English corpora, including Common Voice and LibriSpeech. The alternative uses the English US ARPA acoustic model and dictionary 3.0.0, with a stress-marked ARPAbet inventory and an acoustic model trained on LibriSpeech. The paired comparison changes the acoustic model and dictionary together. The corpus-dictionary and wide-beam runs reuse the baseline acoustic model. [Model specifications](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html)

Both acoustic models use Gaussian mixture model–hidden Markov models (GMM–HMMs). A phone in context is represented by a sequence of hidden states. A state can repeat across frames, which allows a sound to last longer, or transition to another state. A Gaussian mixture gives the likelihood of an observed feature vector for a state. Transition and acoustic scores together let MFA compare different ways of assigning frames to the pronunciation.

MFA and Kaldi compile an utterance-specific weighted finite-state graph. The graph connects the transcript's ordered words to dictionary pronunciations, optional silence and the corresponding HMM transitions. A Viterbi-style dynamic program retains the best path to each reachable state at each frame. Conceptually, it adds the predecessor's score, the transition score and the current frame's acoustic score. Backtracking from a final state recovers the timed path. Beam search prunes partial paths that fall too far behind the best current score. MFA can try a wider retry beam when the first search fails to align an utterance. The saved runs used beam/retry-beam settings of 10/40 and 100/400. [MFA alignment documentation](https://montreal-forced-aligner.readthedocs.io/en/v3.3.9/user_guide/workflows/alignment.html)

## From frames to TextGrids

[align.py](forced_alignment/scripts/align.py) supplies the corpus, compatible models and beam settings to `mfa validate` and `mfa align`. MFA performs feature extraction, graph compilation, decoding and export. It groups consecutive states belonging to each phone occurrence, converts the frame boundaries to seconds and uses the selected pronunciation path to recover word intervals. The resulting Praat TextGrid has separate word and phone tiers. A blank interval can represent an unlabelled gap, such as the quiet tail after the ARPA alignment of “bed”.

[analyze.py](forced_alignment/scripts/analyze.py) reads the exported intervals, checks them against the waveform duration, and matches word labels in order to compare boundary times. It uses words for the cross-model comparison because the phone inventories differ. [visualize.py](forced_alignment/scripts/visualize.py) plots the saved intervals alongside the waveform and Praat spectrogram. The project code prepares, orchestrates and audits the run. The acoustic models are pretrained, and MFA performs the alignment.

## What the results establish

The corpus dictionary reduces OOV coverage gaps from six types and seven tokens to zero, and the seven baseline `spn` intervals disappear. All 243 word labels match the transcript even in the baseline. The wide-beam TextGrids are byte-identical to baseline. US ARPA keeps both boundaries unchanged for 64 of 243 words. For “bed”, the baseline ends at 4.50 s and ARPA ends at 2.06 s. The remaining tail is about 22 dB quieter than the preceding word region. Coverage, model agreement and acoustic evidence describe the outputs. Boundary accuracy requires independent reference annotations.

# Emotion recognition

The [TESS pipeline](emotion_recognition/run.py) runs the three modalities on the same held-out IDs. The grouped split holds entire target words out of training. The two speaker holdouts reserve validation words from the training speaker and test on all words of the other speaker. Those holdouts share vocabulary across train and test. They support a comparison of these two speakers within TESS.

## Speech

A digital waveform samples acoustic pressure at a fixed rate. [features.py](emotion_recognition/models/features.py) converts recordings to mono 16 kHz and conservatively trims each edge. It leaves interword pauses and amplitude variation intact. Each 400-sample Hann window is transformed with a 512-point FFT and advanced by 160 samples. Triangular filters combine nearby power bins on a mel scale, giving 64 pooled energies per frame. Taking their logarithms compresses the dynamic range. A mean and standard deviation per mel bin are estimated from training frames only.

[The speech model](emotion_recognition/models/speech_pipeline/model.py) applies a stride-3 convolution to each 64-band sequence and sends 48-dimensional projected frames to a bidirectional GRU. In a GRU, a reset gate controls how much of the previous state contributes to the candidate state, and an update gate mixes the candidate with the old state. One recurrent pass runs forward and another backward. Their 64-channel outputs are concatenated at each step. Packing restricts recurrence to valid steps and a mask excludes padding from the temporal average. A 128-dimensional speech vector enters the classifier. Recurrent compute grows with the number of projected frames and roughly with the square of recurrent width, which makes the stride useful on CPU.

## Text and joint features

The dataset supplies a carrier phrase and target words. The project reconstructs `Say the word [word]` before tokenization. Emotion, speaker and filename fields stay outside the tokenizer. MiniLM maps subword IDs to learned embeddings, then attention layers contextualize those embeddings using the other valid tokens. Padding is masked in attention and in the final mean. The pooled 384-dimensional vectors are cached once per transcript. The frozen encoder runs in evaluation mode, so stochastic dropout cannot alter repeated vectors. [Text head](emotion_recognition/models/text_pipeline/model.py)

[The fusion model](emotion_recognition/models/fusion_pipeline/model.py) receives the frozen speech embedding from its paired speech-only checkpoint and the cached text vector. Learned projections map them to 48 and 32 dimensions, then concatenate them. A 64-dimensional learned projection is the representation immediately before the seven-class head. The capacity control uses the same speech representation and a comparably sized trainable head. Cross-entropy penalizes low probability assigned to the labelled emotion. AdamW updates the parameters that receive gradients, including the acoustic convolution and GRU in speech-only training. Frozen encoders are evaluated separately, then excluded from head optimization.

The checkpoints are ranked by validation macro F1 with lower validation loss as a tiebreaker. Each run saves held-out probabilities independently of checkpoint selection. [Training loop](emotion_recognition/models/training.py) · [Evaluation and paired comparisons](emotion_recognition/scripts/analyze.py)
