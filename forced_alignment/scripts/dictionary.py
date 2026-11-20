import json
import shutil
from collections import Counter, defaultdict
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from scripts.prepare_data import sha256


RELEASES = {
    "english_mfa.zip": ("acoustic-english_mfa-v3.1.0", "english_mfa.zip", 174044703,
        "2c08bd4f82c3943dd57ac09aeac99dbce17a1e1bfe9fd932c8d95cd13d971068"),
    "english_us_mfa.dict": ("dictionary-english_us_mfa-v3.1.0", "english_us_mfa.dict", 174044841,
        "cbf4ed8cee1338e009b9b387bcb9ee789ce70b98c61ca9bc6ed4f4e63d4812d5"),
    "english_us_mfa_g2p.zip": ("g2p-english_us_mfa-v3.0.0", "english_us_mfa.zip", 153269401,
        "9923b38d59a8b3e3e322f225c52523c2a6248e5ffc9fd89be151ade2dc97cb02"),
    "english_us_arpa.zip": ("acoustic-english_us_arpa-v3.0.0", "english_us_arpa.zip", 153269118,
        "d35ce271ded357d833d2f4b8d1041dc3748b9538567ba13f2c697f4e4126711b"),
    "english_us_arpa.dict": ("dictionary-english_us_arpa-v3.0.0", "english_us_arpa.dict", 153269572,
        "e8c6c7b036ae2b7c78d2768b8dc6b1f9359175b842956d00b48c53c9c332e6b0"),
}


def download_models(folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, (tag, asset, number, digest) in RELEASES.items():
        path = folder / name
        if path.is_file() and sha256(path) == digest:
            continue
        try:
            response = urlopen(f"https://github.com/MontrealCorpusTools/mfa-models/releases/download/{tag}/{asset}", timeout=90)
        except HTTPError:
            response = urlopen(Request(
                f"https://api.github.com/repos/MontrealCorpusTools/mfa-models/releases/assets/{number}?download=1",
                headers={"Accept": "application/octet-stream"}), timeout=90)
        temporary = path.with_suffix(".part")
        with response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary) != digest:
            raise ValueError(f"Release checksum differs: {tag}")
        temporary.replace(path)


def model_metadata(path):
    with ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("meta.json"))
        return json.loads(archive.read(name))


def read_dictionary(path):
    entries = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, _, pronunciation = line.strip().partition("\t")
            if not pronunciation:
                key, pronunciation = line.split(maxsplit=1)
            phones = pronunciation.split("\t")[-1].split()
            entries[key].append((line.strip(), phones))
    return entries


def parts(word, dictionary):
    if word not in dictionary and word.endswith("'s") and "'s" in dictionary:
        return (word[:-2], "'s")
    return (word,)


def missing_words(words, dictionary):
    return [w for w in words if any(piece not in dictionary for piece in parts(w, dictionary))]


def coverage(tokens, dictionary):
    vocabulary = Counter(word for words in tokens.values() for word in words)
    missing = set(missing_words(vocabulary, dictionary))
    return {"types": len(vocabulary), "known_types": len(vocabulary) - len(missing),
            "oov_types": len(missing), "tokens": vocabulary.total(),
            "oov_tokens": sum(vocabulary[w] for w in missing),
            "type_oov_rate": len(missing) / len(vocabulary),
            "token_oov_rate": sum(vocabulary[w] for w in missing) / vocabulary.total(),
            "missing": " ".join(sorted(missing))}


def validate_phones(entries, model):
    allowed = set(model["phone_mapping"])
    unexpected = {phone for variants in entries.values() for _, phones in variants
                  for phone in phones} - allowed
    if unexpected:
        raise ValueError(f"Incompatible dictionary phones: {sorted(unexpected)}")


def build_dictionary(tokens, baseline, models, destination, scratch, run_mfa):
    vocabulary = {word for words in tokens.values() for word in words}
    vocabulary = {piece for w in vocabulary for piece in parts(w, baseline)}
    missing = vocabulary - baseline.keys()
    generated = {}
    if missing:
        input_path = scratch / "oov.txt"
        input_path.write_text("\n".join(sorted(missing)) + "\n", encoding="utf-8")
        output = scratch / "g2p.dict"
        run_mfa(["g2p", "--clean", "--num_jobs", "3", str(input_path),
                 str(models / "english_us_mfa_g2p.zip"), str(output)], scratch / "g2p.log")
        generated = read_dictionary(output)
        if set(generated) != missing:
            raise ValueError(f"G2P coverage mismatch: {sorted(missing ^ generated.keys())}")
    # MFA can split clitics during tokenization. Preserve their official pronunciations.
    clitics = {w for w in baseline if w.startswith("'") or w.endswith("'")}
    support = clitics | (baseline.keys() & {"<unk>", "<eps>", "[bracketed]", "[laughter]"})
    selected = {w: baseline[w] for w in (vocabulary & baseline.keys()) | support}
    selected.update(generated)
    provenance = []
    with destination.open("w", encoding="utf-8") as output:
        for word in sorted(selected):
            for line, phones in selected[word]:
                print(line, file=output)
                if word in vocabulary:
                    provenance.append({"word": word, "phones": " ".join(phones),
                                       "source": "g2p (review)" if word in missing else "official"})
    return selected, provenance
