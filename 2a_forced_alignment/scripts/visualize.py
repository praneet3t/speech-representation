import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from matplotlib.patches import Rectangle

from scripts.analyze import SILENCE


COLORS = ["#286A8A", "#D17B25", "#6C5795", "#34816D"]
LABELS = ["Baseline", "Corpus dict.", "US ARPA", "Wide beam"]


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.png", dpi=300)
    plt.close(fig)


def pipeline(directory):
    fig, ax = plt.subplots(figsize=(7, 2.75), layout="constrained")
    ax.set(xlim=(0, 7), ylim=(0, 2.8))
    ax.axis("off")
    boxes = {
        "audio": (0.68, 2.16, "Audio\n16 kHz PCM"),
        "features": (2.29, 2.16, "MFCCs\n25 ms / 10 ms hop"),
        "model": (4.04, 2.16, "Acoustic model\nGMM–HMM"),
        "text": (0.68, 1.17, "Transcript\nKnown words"),
        "dictionary": (2.29, 1.17, "Lexicon\nG2P for OOV study"),
        "graph": (4.04, 1.17, "Decoding graph\nAllowed phone paths"),
        "decode": (5.95, 1.65, "Constrained search\nBest timed path"),
        "output": (5.95, 0.50, "TextGrid\nWords + phones"),
    }
    nodes = {}
    for key, (x, y, label) in boxes.items():
        nodes[key] = ax.text(x, y, label, ha="center", va="center",
                bbox={"boxstyle": "square,pad=.43", "facecolor": "white", "edgecolor": COLORS[0]})
    fig.canvas.draw()
    for start, end in [("audio", "features"), ("features", "model"), ("text", "dictionary"),
                       ("dictionary", "graph"), ("model", "decode"), ("graph", "decode"), ("decode", "output")]:
        x, y, _ = boxes[start]
        dx, dy, _ = boxes[end]
        ax.annotate("", (dx, dy), (x, y), zorder=0,
                    arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.9,
                                "patchA": nodes[start].get_bbox_patch(),
                                "patchB": nodes[end].get_bbox_patch(), "shrinkA": 1, "shrinkB": 1})
    save(fig, directory, "pipeline")


def draw_intervals(axis, intervals, lo, hi, color, row=0, height=1.0):
    for start, end, label in intervals:
        left, right = max(lo, start), min(hi, end)
        if left >= right:
            continue
        blank = label in SILENCE
        axis.add_patch(Rectangle((left, row), right - left, height, ec="white",
                                 fc="#eeeeee" if blank else color, linewidth=.45))
        if blank:
            continue
        width_points = (right - left) / (hi - lo) * axis.get_window_extent().width
        font_px = axis.figure.dpi / 72 * 7
        if width_points > max(0.72, 0.52 * len(label)) * font_px:
            axis.text((left + right) / 2, row + height / 2, label,
                      color="white", va="center", ha="center", fontsize=7, clip_on=True)


def plot_alignment(directory, wav, name, tiers, center):
    sound = parselmouth.Sound(str(wav))
    # Praat calculates the spectrum directly from the source recording.
    spectrogram = sound.to_spectrogram(window_length=.025, maximum_frequency=8000, time_step=.006)
    power = 10 * np.log10(np.maximum(spectrogram.values, 1e-20))
    ceiling = np.percentile(power, 99.6)
    duration = sound.get_total_duration()
    width = min(4.5, duration)
    lo = max(0, min(center - width / 2, duration - width))
    hi = lo + width
    fig, axes = plt.subplots(4, sharex=True, figsize=(7.2, 3.45),
                             height_ratios=(1, 1.65, .43, .43), layout="constrained")
    waveform, spectrum, words, phones = axes
    waveform.set_title(f"{name}  ·  {lo:.2f}–{hi:.2f} s", loc="left")
    times = sound.xs()
    section = (times >= lo) & (times <= hi)
    waveform.plot(times[section], sound.values[0, section], color=COLORS[0], lw=.55, rasterized=True)
    waveform.set(ylabel="(a) Amplitude", xlim=(lo, hi), ylim=(-1, 1))
    sx = spectrogram.xs()
    region = (sx >= lo - .01) & (sx <= hi + .01)
    spectrum.pcolormesh(sx[region], spectrogram.ys() / 1000, power[:, region],
                        shading="auto", cmap="Greys", vmin=ceiling - 60, vmax=ceiling,
                        rasterized=True)
    spectrum.set(ylabel="(b) kHz", ylim=(0, 8))
    for axis, key, label in [(words, "words", "(c) Word"), (phones, "phones", "(d) Phone")]:
        axis.set(ylim=(0, 1), yticks=[], ylabel=label)
        axis.yaxis.label.set_rotation(0)
        axis.yaxis.label.set_horizontalalignment("right")
        axis.yaxis.set_label_coords(-.018, .5)
        draw_intervals(axis, tiers[key], lo, hi, COLORS[0])
    phones.set_xlabel("Time (s)")
    save(fig, directory, name)

def plot_corpus(results, directory, data):
    with (results / "dictionary_coverage.csv").open(encoding="utf-8", newline="") as stream:
        cover = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    rows = data["inventory"]
    short_names = [r["utterance"] if r["utterance"].startswith("F2") else
                   r["utterance"].replace("ISLE_SESS0131_BLOCKD02_", "ISLE ").replace("_sprt1", "")
                   for r in rows]
    axes[0, 0].bar(short_names, [float(r["duration_s"]) for r in rows], color=COLORS[0])
    axes[0, 0].set(title="(a) Recording length", ylabel="Duration (s)")
    axes[0, 1].bar(short_names, [int(r["words"]) for r in rows], color=COLORS[0])
    axes[0, 1].set(title=f"(b) Transcript size ({cover[0]['types']} types)", ylabel="Word tokens")
    for axis in axes[0]:
        axis.tick_params(axis="x", labelrotation=32, labelsize=7)
        plt.setp(axis.get_xticklabels(), ha="right", rotation_mode="anchor")
    x = np.arange(3)
    axes[1, 0].bar(x - .17, [100 * float(r["type_oov_rate"]) for r in cover], .34,
                   color=COLORS[0], label="Types")
    axes[1, 0].bar(x + .17, [100 * float(r["token_oov_rate"]) for r in cover], .34,
                   color=COLORS[1], label="Tokens")
    axes[1, 0].set(title="(c) Dictionary coverage", ylabel="Out of vocabulary (%)", xticks=x,
                   xticklabels=[label.replace(" ", "\n") for label in LABELS[:3]])
    axes[1, 0].legend(frameon=False, fontsize=8)
    runtime = axes[1, 1]
    configs = list(data["aggregate"].values())
    runtime.bar(np.arange(4), [r["alignment_seconds"] for r in configs], color=COLORS)
    runtime.set(title="(d) Alignment wall time", ylabel="Time (s)",
                xticks=np.arange(4), xticklabels=[label.replace(" ", "\n") for label in LABELS])
    runtime.tick_params(axis="x", labelsize=7.5)
    for i, row in enumerate(configs):
        runtime.text(i, row["alignment_seconds"], f'{row["real_time_factor"]:.2f}×',
                     ha="center", va="bottom", fontsize=8)
    runtime.set_ylim(0, 1.2 * max(r["alignment_seconds"] for r in configs))
    save(fig, directory, "corpus_and_runtime")


def plot_distributions(directory, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), layout="constrained")
    grids_by_config = data["grids"]
    same_beams = grids_by_config["baseline"] == grids_by_config["decoding_sensitivity"]
    for i, (config, grids) in enumerate(grids_by_config.items()):
        if same_beams and config == "decoding_sensitivity":
            continue
        curve_label = "Baseline / wide beam" if same_beams and config == "baseline" else LABELS[i]
        for axis, tier, edges in [(axes[0], 0, np.arange(0, 601, 20)), (axes[1], 1, np.arange(0, 401, 10))]:
            samples = [1000 * (end - start) for grid in grids.values() for start, end, label in grid[tier]
                       if label not in {"<unk>", "spn", "<oov>"}]
            if samples:
                counts, _ = np.histogram(np.minimum(samples, edges[-1]), edges)
                axis.stairs(counts / len(samples) * 100, edges, label=curve_label, color=COLORS[i], lw=1.3)
    axes[0].set(title="(a) Word durations", ylabel="Intervals (%)", xlabel="Duration (ms)",
                xticks=[0, 100, 200, 300, 400, 500, 580],
                xticklabels=["0", "100", "200", "300", "400", "500", "≥580"])
    axes[1].set(title="(b) Phone durations", xlabel="Duration (ms)",
                xticks=[0, 100, 200, 300, 390], xticklabels=["0", "100", "200", "300", "≥390"])
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, directory, "durations")


def plot_displacement(directory, data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    maximum = max((max(r["start_abs_ms"], r["end_abs_ms"]) for r in data["boundaries"]), default=10)
    for i, config in enumerate(list(data["aggregate"])[1:], 1):
        rows = [r for r in data["boundaries"] if r["configuration"] == config]
        for axis, field in zip(axes, ("start_abs_ms", "end_abs_ms")):
            values = np.sort([r[field] for r in rows])
            if len(values):
                axis.step(np.r_[0, values, maximum], np.r_[np.arange(len(values) + 1) / len(values) * 100, 100],
                          where="post", label=LABELS[i], color=COLORS[i], lw=1.3,
                          ls=("--" if i == 3 else "-"))
    for axis, title in zip(axes, ("(a) Word start", "(b) Word end")):
        axis.set(title=title, xlabel="Absolute displacement (ms)", xlim=(0, None), ylim=(0, 104))
        axis.set_xscale("symlog", linthresh=10)
    axes[0].set_ylabel("Matched words (%)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, directory, "boundary_displacement")


def plot_cases(directory, data):
    cases = data["observations"]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), figsize=(7.2, 1.6 * len(cases)), squeeze=False, layout="constrained")
    for index, (ax, case) in enumerate(zip(axes.ravel(), cases)):
        start, end = case["baseline_start_s"], case["baseline_end_s"]
        lo = max(0, min(start, case.get("comparison_start_s", start)) - .26)
        hi = max(end, case.get("comparison_end_s", end)) + .28
        configurations = ["baseline", "corpus_dictionary" if case["configuration"] == "baseline"
                          else case["configuration"]]
        tier = 1 if case["case"] == "unknown pronunciation" else 0
        for i, config in enumerate(reversed(configurations)):
            grids = data["grids"][config]
            if case["utterance"] not in grids:
                continue
            grid = grids[case["utterance"]]
            intervals = grid[2]["phones"] if tier else grid[0]
            draw_intervals(ax, intervals, lo, hi, COLORS[list(data["aggregate"]).index(config)], i, .84)
        ax.axvline(start, color="#222222", lw=.7, ls="--")
        ax.axvline(end, color="#222222", lw=.7, ls="--")
        ax.set(xlim=(lo, hi), ylim=(0, 2), yticks=(.42, 1.42),
               yticklabels=[LABELS[list(data["aggregate"]).index(c)] for c in reversed(configurations)],
               title=f"({chr(97 + index)}) {case['utterance']} · {case['case']}")
    axes[-1, 0].set_xlabel("Time (s)")
    save(fig, directory, "boundary_examples")


def visualize(results, corpus, data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    directory = results / "plots"
    directory.mkdir(exist_ok=True)
    pages = directory / "alignments"
    pages.mkdir(exist_ok=True)
    for path in pages.iterdir():
        if path.suffix in {".pdf", ".png"}:
            path.unlink()
    pipeline(directory)
    plot_corpus(results, directory, data)
    plot_distributions(directory, data)
    plot_displacement(directory, data)
    plot_cases(directory, data)
    for name, (_, _, tiers, _) in data["grids"]["baseline"].items():
        rows = [r for r in data["boundaries"] if r["utterance"] == name
                and r["configuration"] == "alternate_model"]
        example = max(rows, key=lambda r: max(r["start_abs_ms"], r["end_abs_ms"]), default=None)
        center = (example["baseline_start_s"] + example["baseline_end_s"]) / 2 if example else 2.25
        plot_alignment(pages, corpus / name / f"{name}.wav", name, tiers, center)
