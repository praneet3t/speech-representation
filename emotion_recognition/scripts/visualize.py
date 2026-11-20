from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from models.dataset import EMOTIONS, PROTOCOLS, SEEDS
from models.features import HOP, SAMPLE_RATE, WINDOW
from scripts.analyze import MAIN


PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#555555")
MODELS = dict(zip(MAIN, PALETTE[:3]))
NAMES = {"speech": "Speech", "text": "Text", "fusion": "Fusion", "control": "Speech control"}
PROTOCOL_NAMES = ("Word-disjoint", "Hold out OAF", "Hold out YAF")
SHORT = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.png", dpi=300)
    plt.close(fig)


def architecture(directory):
    fig, axis = plt.subplots(figsize=(7.4, 3.1), layout="constrained")
    axis.set(xlim=(0, 7.4), ylim=(0, 3.25))
    axis.axis("off")
    rows = [
        ("Speech", ["16 kHz audio\nedge trim", "Log-Mel\nB × T × 64", "Stride-3 conv\nB × T′ × 48",
                    "Packed BiGRU\nmasked mean\nB × 128", "MLP head\nB × 7"], PALETTE[0]),
        ("Text", ["Carrier phrase\ntokenizer", "Token embeddings\nB × L × 384",
                  "MiniLM layers\nB × L × 384", "Attention-mask\nmean · B × 384", "MLP head\nB × 7"], PALETTE[1]),
        ("Fusion", ["Speech B × 128\nText B × 384", "Project · concat\nB × (48 + 32)",
                    "Fusion layer\nB × 64", "Classifier\nB × 7"], PALETTE[2]),
    ]
    for i, (title, labels, color) in enumerate(rows):
        y = 2.70 - i * 1.02
        axis.text(.07, y + .4, title, color=color, weight="bold", va="center")
        for j, label in enumerate(labels):
            spacing = 1.50 if i < 2 else 2.0
            x = j * spacing + .045
            frozen = (i == 1 and 0 < j < 4) or (i == 2 and j == 0)
            trainable = (i == 0 and j >= 2) or (i == 1 and j == 4) or (i == 2 and j >= 1)
            axis.add_patch(FancyBboxPatch((x, y - .27), 1.29, .54, boxstyle="square,pad=0",
                                          facecolor="#ebebeb" if frozen else "white",
                                          edgecolor=color if trainable else "#888888", linewidth=.9))
            axis.text(x + .645, y, label, ha="center", va="center", size=7.1)
            if j < len(labels) - 1:
                axis.annotate("", (x + spacing - .01, y), (x + 1.3, y),
                              arrowprops={"arrowstyle": "->", "lw": .7, "color": "#555555"})
    axis.text(.05, .08, "Grey fill: frozen encoder   ·   Colored border: trained module",
              fontsize=7, color="#444444")
    save(fig, directory, "architecture")


def dataset_plot(directory, audit):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), layout="constrained")
    counts = [list(audit["by_speaker_emotion"][speaker][e] for e in EMOTIONS) for speaker in ("OAF", "YAF")]
    x = np.arange(len(EMOTIONS))
    axes[0].bar(x, counts[0], color=PALETTE[0], label="OAF")
    axes[0].bar(x, counts[1], bottom=counts[0], color=PALETTE[3], label="YAF")
    axes[0].set(title="(a) Corpus composition", ylabel="Unique recordings",
                xticks=x, xticklabels=SHORT, ylim=(0, 480))
    axes[0].tick_params(axis="x", rotation=34)
    axes[0].legend(ncols=2, loc="upper center", frameon=False, fontsize=7)
    first = next(iter(audit["conditional_probabilities"].values()))
    axes[1].bar(x, [100 * first[e] for e in EMOTIONS], color=PALETTE[:len(x)])
    axes[1].set(title="(b) Emotion given any transcript", ylabel="Conditional frequency (%)",
                xticks=x, xticklabels=SHORT, ylim=(0, 19))
    axes[1].tick_params(axis="x", rotation=34)
    save(fig, directory, "dataset")


def primary_plot(directory, summary):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), layout="constrained")
    word = {r["model"]: r for r in summary if r["protocol"] == "word"}
    for axis, metric, title in zip(axes, ("accuracy", "macro_f1"), ("(a) Accuracy", "(b) Macro F1"), strict=True):
        axis.bar(np.arange(len(MAIN)), [100 * word[m][metric + "_mean"] for m in MAIN],
                 yerr=[100 * word[m][metric + "_sd"] for m in MAIN], capsize=3,
                 color=[MODELS[m] for m in MAIN])
        axis.set(title=title, ylabel="Score (%)", ylim=(0, 108), xticks=np.arange(len(MAIN)),
                 xticklabels=[NAMES[m] for m in MAIN])
        if metric == "accuracy":
            axis.axhline(100 / len(EMOTIONS), ls="--", color="#555555", lw=.9)
    save(fig, directory, "primary_performance")


def per_emotion_plot(directory, per_class):
    fig, axis = plt.subplots(figsize=(7.1, 2.7), layout="constrained")
    x = np.arange(len(EMOTIONS))
    for i, model in enumerate(MAIN):
        values = [[r["f1"] * 100 for r in per_class if r["protocol"] == "word" and r["model"] == model
                   and r["emotion"] == e] for e in EMOTIONS]
        axis.bar(x + .25 * (i - 1), np.mean(values, axis=1), .24,
                 yerr=np.std(values, axis=1, ddof=1), color=MODELS[model], capsize=2, label=NAMES[model])
    axis.set(ylabel="F1 (%)", xticks=x, xticklabels=SHORT, ylim=(0, 118))
    axis.legend(frameon=False, ncols=3, loc="upper center")
    save(fig, directory, "per_emotion")


def confusion_plot(directory, confusion):
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), layout="constrained")
    ticks = ("Ang", "Dis", "Fea", "Hap", "Neu", "Sad", "Sur")
    for i, (axis, model) in enumerate(zip(axes, MAIN, strict=True)):
        matrix = np.zeros((len(EMOTIONS), len(EMOTIONS)))
        for r in confusion:
            if r["protocol"] == "word" and r["model"] == model:
                matrix[EMOTIONS.index(r["true"]), EMOTIONS.index(r["predicted"])] += r["count"]
        matrix = 100 * matrix / matrix.sum(axis=1, keepdims=True)
        table = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        for y, x in np.argwhere(matrix > 0):
            value = matrix[y, x]
            label = "100" if value == 100 else f"{value:.1f}"
            axis.text(x, y, label, fontsize=6.3, ha="center", va="center",
                      color="white" if matrix[y, x] > 50 else "#222222")
        axis.set(title=f"({chr(97 + i)}) {NAMES[model]}", xlabel="Predicted",
                 xticks=range(len(ticks)), yticks=range(len(ticks)), xticklabels=ticks, yticklabels=ticks)
        axis.tick_params(axis="x", rotation=50, labelsize=7)
        axis.tick_params(axis="y", labelsize=7)
    axes[0].set_ylabel("True emotion")
    fig.colorbar(table, ax=axes, shrink=.72, label="Row percentage")
    save(fig, directory, "confusions")


def learning_curves(directory, histories):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.1), sharey="row", layout="constrained")
    for i, model in enumerate(MAIN):
        rows = [r for r in histories if r["protocol"] == "word" and r["seed"] == SEEDS[0] and r["model"] == model]
        epochs = [int(r["epoch"]) for r in rows]
        for j, metric in enumerate(("loss", "accuracy")):
            axis = axes[j, i]
            for part, color in [("train", PALETTE[0]), ("val", PALETTE[1])]:
                axis.plot(epochs, [float(r[f"{part}_{metric}"]) for r in rows], lw=1.2,
                          color=color, label="Train" if part == "train" else "Validation")
            if j == 0:
                axis.set_title(NAMES[model])
            else:
                axis.set(xlabel="Epoch", ylim=(0, 1.05))
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[1, 0].set_ylabel("Accuracy")
    axes[0, 0].legend(frameon=False, fontsize=7)
    save(fig, directory, "learning_curves")


def projection_plot(directory, projections):
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.7), layout="constrained")
    for i, model in enumerate(MAIN):
        rows = [r for r in projections if r["model"] == model]
        for j, method in enumerate(("pca", "tsne")):
            axis = axes[j, i]
            if model == "text":
                groups = Counter((r[method + "_x"], r[method + "_y"]) for r in rows)
                coords = list(groups)
                axis.scatter(*np.asarray(coords).T, s=23, facecolors="none", edgecolors="#555555", lw=.65)
            else:
                for emotion, color in zip(EMOTIONS, PALETTE, strict=True):
                    points = [(r[method + "_x"], r[method + "_y"]) for r in rows if r["emotion"] == emotion]
                    axis.scatter(*np.array(points).T, color=color, s=8, alpha=.65, label=emotion.capitalize())
            axis.set(xticks=[], yticks=[])
            if j == 0:
                axis.set_title(NAMES[model])
    axes[0, 0].set_ylabel("PCA")
    axes[1, 0].set_ylabel("t-SNE")
    axes[1, 1].set_xlabel("Each open circle spans all seven emotions", fontsize=7)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=7, fontsize=7)
    save(fig, directory, "representations")


def speaker_plot(directory, summary):
    fig, axis = plt.subplots(figsize=(6.9, 2.7), layout="constrained")
    x = np.arange(2)
    for i, model in enumerate(MAIN):
        rows = [next(r for r in summary if r["protocol"] == p and r["model"] == model) for p in PROTOCOLS[1:]]
        axis.bar(x + .24 * (i - 1), [100 * r["accuracy_mean"] for r in rows], .22,
                 yerr=[100 * r["accuracy_sd"] for r in rows], capsize=3,
                 color=MODELS[model], label=NAMES[model])
    axis.set(ylabel="Accuracy (%)", xticks=x, xticklabels=PROTOCOL_NAMES[1:], ylim=(0, 110))
    axis.legend(frameon=False, ncols=3, loc="upper center")
    save(fig, directory, "speaker_holdout")


def ablation_plot(directory, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=True, layout="constrained")
    categories = ("fusion", "control", "mask_text", "mask_speech", "permute_text")
    names = ("Fusion", "Speech ctrl.", "Mask text", "Mask speech", "Permute text")
    for axis, p, title in zip(axes, PROTOCOLS, PROTOCOL_NAMES, strict=True):
        values = [[np.mean([r["accuracy"] * 100 for r in metrics if r["protocol"] == p and r["seed"] == seed and
                           (r["model"].startswith("permute_text_") if v == "permute_text" else r["model"] == v)])
                   for seed in SEEDS] for v in categories]
        axis.bar(range(len(categories)), np.mean(values, axis=1), yerr=np.std(values, axis=1, ddof=1),
                 color=[PALETTE[2], PALETTE[0], "#8073AC", "#B35806", "#666666"], capsize=2)
        axis.set(title=title, xticks=range(len(categories)), xticklabels=names, ylim=(0, 108))
        axis.tick_params(axis="x", rotation=37, labelsize=7)
    axes[0].set_ylabel("Accuracy (%)")
    save(fig, directory, "ablations")


def failure_plot(directory, cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.2, 1.45 * len(cases)), squeeze=False, layout="constrained")
    for i, ((meta, signal, mel), (wave, spec)) in enumerate(zip(cases, axes, strict=True)):
        limit = len(signal) / SAMPLE_RATE
        wave.plot(np.arange(len(signal)) / SAMPLE_RATE, signal, color=PALETTE[0], lw=.65, rasterized=True)
        wave.set(ylabel="Amplitude", xlim=(0, limit), ylim=(-1, 1),
                 title=f"{meta['id']}  ·  {meta['protocol'].replace('_', ' ')}")
        power_db = 10 * mel.T / np.log(10)
        edge = np.percentile(power_db, 99)
        first_edge = (WINDOW - HOP) / (2 * SAMPLE_RATE)
        spec.imshow(power_db, aspect="auto", origin="lower", cmap="cividis", vmin=edge - 58, vmax=edge,
                    extent=(first_edge, first_edge + len(mel) * HOP / SAMPLE_RATE, 0, mel.shape[1]),
                    rasterized=True)
        spec.set(xlim=(0, limit), ylabel="Mel bin",
                 title=f"{meta['true']}  |  speech: {meta['speech_prediction']}  |  fusion: {meta['fusion_prediction']}")
        if i == len(cases) - 1:
            wave.set_xlabel("Time after edge trim (s)")
            spec.set_xlabel("Time after edge trim (s)")
    save(fig, directory, "failure_cases")


def figures(output, audit, summary, metrics, per_class, histories, confusion, projections, failure_cases):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    architecture(directory)
    dataset_plot(directory, audit)
    primary_plot(directory, summary)
    per_emotion_plot(directory, per_class)
    confusion_plot(directory, confusion)
    learning_curves(directory, histories)
    projection_plot(directory, projections)
    speaker_plot(directory, summary)
    ablation_plot(directory, metrics)
    failure_plot(directory, failure_cases)
