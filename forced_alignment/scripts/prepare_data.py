import hashlib
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def index_files(directory, suffix):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() == suffix:
            key = path.stem.casefold()
            if key in files:
                raise ValueError(f"Duplicate identifier: {files[key].name}, {path.name}")
            files[key] = path
    return files


def normalize(text):
    text = unicodedata.normalize("NFC", text).replace("’", "'")
    # Read the two initialisms as letter names, keeping the possessive suffix.
    text = text.replace("S.J.C.", "S J C").replace("WBUR", "W B U R")
    text = re.sub(r"\bde-politicize\b", "depoliticize", text, flags=re.I)
    # Hyphenated numbers in this corpus are already written out.
    return " ".join(re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower()))


def discover_inputs(data):
    roots = [p for p in (data, *sorted(data.iterdir()))
             if (p / "wav").is_dir() and (p / "transcripts").is_dir()] if data.is_dir() else []
    if len(roots) != 1:
        raise ValueError("Extract the supplied zip under forced_alignment/data, or pass --wav and --transcripts.")
    return roots[0] / "wav", roots[0] / "transcripts"


def prepare(wav_dir, text_dir, corpus):
    wavs = index_files(wav_dir, ".wav")
    texts = index_files(text_dir, ".txt")
    rows, tokens, seen_audio, issues = [], {}, {}, []
    if not wavs and not texts:
        raise ValueError("No WAVs or transcripts found.")
    for key in sorted(wavs.keys() | texts.keys()):
        wav, txt = wavs.get(key), texts.get(key)
        name = (wav or txt).stem
        row = {"utterance": name, "audio": wav.name if wav else "",
               "transcript": txt.name if txt else "",
               "audio_sha256": sha256(wav) if wav else "",
               "transcript_sha256": sha256(txt) if txt else ""}
        rows.append(row)
        try:
            if wav is None or txt is None:
                raise ValueError("unmatched file")
            original = txt.read_text(encoding="utf-8-sig")
            normalized = normalize(original)
            if not normalized or any(c.isdigit() for c in normalized) or any(c in original for c in "[]<>"):
                raise ValueError("empty transcript or unsupported numeral/annotation")
            target = corpus / name
            target.mkdir(parents=True, exist_ok=True)
            info = sf.info(wav)
            if info.channels != 1 or info.samplerate != 16000 or info.subtype != "PCM_16":
                raise ValueError(f"expected mono 16 kHz PCM16, got {info}")
            audio, rate = sf.read(wav, dtype="int16")
            if len(audio) != info.frames or not len(audio):
                raise ValueError("empty or truncated recording")
            pcm_hash = hashlib.sha256(np.asarray(audio, dtype="<i2").tobytes()).hexdigest()
            row.update(sample_rate_hz=rate, channels=info.channels, frames=info.frames,
                       duration_s=info.duration, words=len(normalized.split()), speaker="",
                       duplicate_of=seen_audio.get(pcm_hash, ""), source_format=info.format,
                       pcm_sha256=pcm_hash, status="paired")
            seen_audio.setdefault(pcm_hash, name)
            prepared_wav = target / f"{name}.wav"
            sf.write(prepared_wav, audio, rate, subtype="PCM_16")
            assert np.array_equal(sf.read(prepared_wav, dtype="int16")[0], audio)
            lab = target / f"{name}.lab"
            lab.write_text(normalized + "\n", encoding="utf-8")
            row["normalized_sha256"] = sha256(lab)
            saved = corpus.parent / "original_transcripts"
            saved.mkdir(exist_ok=True)
            shutil.copyfile(txt, saved / txt.name)
            tokens[name] = normalized.split()
        except (OSError, RuntimeError, ValueError) as error:
            row["status"] = str(error)
            issues.append(name)
    return rows, tokens, issues
