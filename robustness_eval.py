from pathlib import Path
import re
import string

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation import calculate_metrics
from pipeline import load_models, build_doc_matrix
from features import split_sentences, is_scorable


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "evaluation_final" / "robustness"
OUTPUT.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
MAX_DOCS = 10000
SEED = 5
BATCH_SIZE = 100

models = load_models(str(ROOT / "models"))
models["rfr"]["model"].n_jobs = 1

punctuation_table = str.maketrans(
    {character: " " for character in string.punctuation}
)

TRANSFORMS = {
    "lowercase": lambda text: text.lower(),
    "remove_punctuation": lambda text: text.translate(
        punctuation_table
    ),
    "merge_lines": lambda text: re.sub(
        r"[\r\n]+", " ", text
    ),
}


def eligible(text):
    return any(
        is_scorable(sentence)
        for sentence in split_sentences(text)
    )


def predict_batches(texts):
    scores = np.full(len(texts), np.nan)

    valid = [
        index
        for index, text in enumerate(texts)
        if eligible(text)
    ]

    for offset in range(0, len(valid), BATCH_SIZE):
        indices = valid[offset:offset + BATCH_SIZE]
        batch = [texts[index] for index in indices]

        X, kept = build_doc_matrix(
            batch,
            models["clf"],
            models["gmm"],
            n_jobs=1,
        )

        predictions = models["rfr"]["model"].predict(X)

        for local_index, score in zip(kept, predictions):
            scores[indices[int(local_index)]] = score

        print(
            f"  {min(offset + BATCH_SIZE, len(valid)):,}"
            f"/{len(valid):,}documents processed",
            flush=True,
        )

    finite = scores[np.isfinite(scores)]

    if ((finite < 0) | (finite > 1)).any():
        raise ValueError("Predicted scores must be between 0 and 1.")

    return scores


rows = []

for dataset, filename in [
    ("test", "test.parquet"),
    ("holdout_2026", "holdout_2026.parquet"),
]:
    path = ROOT / "data" / "splits" / filename

    df = pd.read_parquet(
        path, columns=["text", "is_ai"]
    )

    if (
        df.empty
        or df["text"].isna().any()
        or not df["text"].map(
            lambda value: isinstance(value, str)
        ).all()
        or not df["is_ai"].isin([0, 1]).all()
    ):
        raise ValueError(f"{filename}: Invalid text or labels.")

    df["text"] = df["text"].str.strip()

    conflicts = df.groupby("text")["is_ai"].nunique().gt(1)

    if conflicts.any():
        raise ValueError(f"{filename}: Conflicting labels for the same text.")

    df = df.drop_duplicates("text").reset_index(drop=True)

    if len(df) > MAX_DOCS:
        df = df.sample(
            MAX_DOCS, random_state=SEED
        ).reset_index(drop=True)

    texts = df["text"].tolist()
    labels = df["is_ai"].to_numpy(dtype=int)

    print(f"\n{dataset}: 원본 평가", flush=True)
    original_scores = predict_batches(texts)

    for condition, transform in TRANSFORMS.items():
        print(
            f"\n{dataset}: {condition} Evaluating original texts",
            flush=True,
        )

        changed_texts = [transform(text) for text in texts]
        changed_scores = predict_batches(changed_texts)

        paired = (
            np.isfinite(original_scores)
            & np.isfinite(changed_scores)
        )

        pair_n = int(paired.sum())

        if pair_n == 0:
            raise ValueError(
                f"{dataset}/{condition}: No paired scorable documents."
            )

        y = labels[paired]
        original = original_scores[paired]
        changed = changed_scores[paired]

        original_metrics = calculate_metrics(
            y, original, THRESHOLD
        )
        changed_metrics = calculate_metrics(
            y, changed, THRESHOLD
        )

        original_pred = original > THRESHOLD
        changed_pred = changed > THRESHOLD

        rows.append({
            "dataset": dataset,
            "condition": condition,
            "sampled_n": len(df),
            "paired_n": pair_n,
            "pair_coverage": pair_n / len(df),
            "original_unscorable_n": int(
                (~np.isfinite(original_scores)).sum()
            ),
            "transformed_unscorable_n": int(
                (~np.isfinite(changed_scores)).sum()
            ),
            "original_f1": original_metrics["f1"],
            "f1_change": (
                changed_metrics["f1"]
                - original_metrics["f1"]
            ),
            "prediction_change_rate": float(
                (original_pred != changed_pred).mean()
            ),
            "mean_absolute_score_change": float(
                np.abs(original - changed).mean()
            ),
            **changed_metrics,
        })

        details = pd.DataFrame({
            "original_text": df.loc[paired, "text"].to_numpy(),
            "transformed_text": np.asarray(
                changed_texts, dtype=object
            )[paired],
            "is_ai": y,
            "original_score": original,
            "transformed_score": changed,
            "original_prediction": original_pred.astype(int),
            "transformed_prediction": changed_pred.astype(int),
            "prediction_changed": (
                original_pred != changed_pred
            ),
        })

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT / "robustness_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True,
    layout="constrained",
)

for ax, dataset in zip(axes, ["test", "holdout_2026"]):
    part = results[results["dataset"] == dataset]
    x = np.arange(len(part))

    for offset, column, label, color in [
        (-0.18, "original_f1", "Original", "#4C78A8"),
        (0.18, "f1", "Transformed", "#E45756"),
    ]:
        bars = ax.bar(
            x + offset,
            part[column],
            width=0.36,
            label=label,
            color=color,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3)

    ax.set_xticks(
        x,
        ["Lowercase", "Remove\npunctuation", "Merge\nlines"],
    )
    ax.set_title(dataset)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

axes[0].set_ylabel("F1-score")
axes[0].legend()

fig.suptitle("Robustness evaluation: matched documents")

fig.savefig(
    OUTPUT / "robustness_f1.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\n=== Robustness evaluation results ===")

print(
    results[[
        "dataset",
        "condition",
        "paired_n",
        "original_f1",
        "f1",
        "f1_change",
        "prediction_change_rate",
    ]].round(4).to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT}")