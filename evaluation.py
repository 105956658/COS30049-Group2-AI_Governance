"""
AI 생성 텍스트 탐지 모델 평가

설치:
    pip install pandas numpy scikit-learn matplotlib joblib pyarrow

실행:
    python evaluation.py

먼저 1,000개만 평가:
    python evaluation.py --max-docs 1000

필요한 데이터 컬럼:
    text: 텍스트
    is_ai: Human=0, AI=1

기존 모델을 불러오며 재학습하지 않습니다.
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from pipeline import load_models, build_doc_matrix
from features import split_sentences, is_scorable, doc_feature_names


# ---------------------------------------------------------
# 1. 평가 지표
# ---------------------------------------------------------

def calculate_metrics(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    # 팀의 기존 regressor.py와 같은 기준
    predictions = (scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    both_classes = len(np.unique(y_true)) == 2

    return {
        "n": len(y_true),
        "human_n": int((y_true == 0).sum()),
        "ai_n": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": (
            balanced_accuracy_score(y_true, predictions)
            if both_classes else np.nan
        ),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "roc_auc": (
            roc_auc_score(y_true, scores)
            if both_classes else np.nan
        ),
        "average_precision": (
            average_precision_score(y_true, scores)
            if both_classes else np.nan
        ),
        "brier": brier_score_loss(y_true, scores),
        "mae": mean_absolute_error(y_true, scores),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, scores))
        ),
        "false_positive_rate": (
            fp / (fp + tn) if fp + tn else np.nan
        ),
        "false_negative_rate": (
            fn / (fn + tp) if fn + tp else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------
# 2. Bootstrap으로 F1 변동성 추정
# ---------------------------------------------------------

def bootstrap_f1(y_true, scores, threshold, repetitions, seed):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)

    groups = [
        np.flatnonzero(y_true == 0),
        np.flatnonzero(y_true == 1),
    ]

    if repetitions == 0 or any(len(g) == 0 for g in groups):
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    predictions = scores > threshold
    values = []

    for _ in range(repetitions):
        # 클래스 비율을 유지하며 문서를 복원 추출
        indices = np.concatenate([
            rng.choice(g, len(g), replace=True)
            for g in groups
        ])

        values.append(
            f1_score(
                y_true[indices],
                predictions[indices],
                zero_division=0,
            )
        )

    lower, upper = np.percentile(values, [2.5, 97.5])

    return float(lower), float(upper)


# ---------------------------------------------------------
# 3. 데이터 검증
# ---------------------------------------------------------

def validate_data(df, path):
    if not {"text", "is_ai"}.issubset(df.columns):
        raise ValueError(
            f"{path}: text와 is_ai 컬럼이 필요합니다."
        )

    if df.empty:
        raise ValueError(f"{path}: 데이터가 비어 있습니다.")

    if (
        df["text"].isna().any()
        or not df["text"].map(
            lambda value: isinstance(value, str)
        ).all()
    ):
        raise ValueError(
            f"{path}: text에 결측값이나 문자열이 아닌 값이 있습니다."
        )

    if (
        df["is_ai"].isna().any()
        or not df["is_ai"].isin([0, 1]).all()
    ):
        raise ValueError(
            f"{path}: is_ai는 0 또는 1이어야 합니다."
        )

    normalized_text = df["text"].str.strip()

    # 같은 글에 다른 라벨이 있으면 중단하고 확인
    conflicting_labels = (
        df.assign(normalized_text=normalized_text)
        .groupby("normalized_text")["is_ai"]
        .nunique()
        .gt(1)
        .any()
    )

    if conflicting_labels:
        raise ValueError(
            f"{path}: 동일한 글에 서로 다른 정답 라벨이 있습니다."
        )

    df = df.copy()
    df["original_row"] = np.arange(len(df))

    duplicate_mask = normalized_text.duplicated()
    duplicate_count = int(duplicate_mask.sum())

    df = df.loc[~duplicate_mask].reset_index(drop=True)

    return df, duplicate_count


# ---------------------------------------------------------
# 4. 그래프
# ---------------------------------------------------------

def save_plots(result, dataset_name, output_dir):
    y_true = result["is_ai"].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        result["prediction"],
        labels=[0, 1],
        display_labels=["Human", "AI"],
        ax=ax,
        colorbar=False,
    )

    ax.set_title(f"{dataset_name}: final model")
    fig.tight_layout()
    fig.savefig(
        output_dir / f"{dataset_name}_confusion_matrix.png",
        dpi=180,
    )
    plt.close(fig)

    if len(np.unique(y_true)) != 2:
        return

    for plot_name, display in [
        ("roc", RocCurveDisplay),
        ("precision_recall", PrecisionRecallDisplay),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))

        for label, column in [
            ("Final model", "final_score"),
            ("Sentence mean", "baseline_score"),
        ]:
            display.from_predictions(
                y_true,
                result[column],
                name=label,
                ax=ax,
            )

        ax.set_title(dataset_name)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{dataset_name}_{plot_name}.png",
            dpi=180,
        )
        plt.close(fig)


# ---------------------------------------------------------
# 5. 오탐/미탐 사례
# ---------------------------------------------------------

def save_error_examples(result, dataset_name, output_dir, seed):
    examples = []

    for error_type in ["false_positive", "false_negative"]:
        errors = result.loc[
            result["error_type"] == error_type
        ]

        if not errors.empty:
            selected = errors.sample(
                min(5, len(errors)),
                random_state=seed,
            ).copy()

            # 사례를 직접 읽고 관찰 내용을 작성
            selected["manual_observation"] = ""
            examples.append(selected)

    if examples:
        review = pd.concat(examples, ignore_index=True)
    else:
        review = result.iloc[:0].copy()
        review["manual_observation"] = ""

    review.to_csv(
        output_dir / f"{dataset_name}_error_review.csv",
        index=False,
    )


# ---------------------------------------------------------
# 6. 데이터셋 평가
# ---------------------------------------------------------

def evaluate_dataset(path, name, models, args, output_dir):
    raw = pd.read_parquet(path)
    df, duplicates = validate_data(raw, path)

    if args.max_docs and len(df) > args.max_docs:
        df = df.sample(
            args.max_docs,
            random_state=args.seed,
        ).reset_index(drop=True)

    # 팀 모델은 4단어 이상의 문장만 처리
    eligible = df["text"].map(
        lambda text: any(
            is_scorable(sentence)
            for sentence in split_sentences(text)
        )
    )

    excluded = df.loc[~eligible].copy()
    excluded["reason"] = (
        "No sentence with at least 4 whitespace-separated words"
    )
    excluded.to_csv(
        output_dir / f"{name}_excluded.csv",
        index=False,
    )

    coverage = {
        "dataset": name,
        "raw_n": len(raw),
        "duplicate_rows_removed": duplicates,
        "sampled_n": len(df),
        "scored_n": int(eligible.sum()),
        "excluded_n": int((~eligible).sum()),
        "coverage": float(eligible.mean()),
    }

    if not eligible.any():
        return [], [], coverage

    usable = df.loc[eligible].reset_index(drop=True)

    start = time.perf_counter()

    matrices = []
    kept_rows = []

    batch_size = 100

    for offset in range(0, len(usable), batch_size):
        batch = usable.iloc[offset:offset + batch_size]

        batch_X, batch_kept = build_doc_matrix(
            batch["text"].tolist(),
            models["clf"],
            models["gmm"],
            n_jobs=args.n_jobs,
        )

        matrices.append(batch_X)
        kept_rows.extend(
            offset + int(index) for index in batch_kept
        )

        print(
            f"{name}: {min(offset + batch_size, len(usable)):,}"
            f"/{len(usable):,}개 처리",
            flush=True,
        )

    X = np.vstack(matrices)
    kept_indices = np.asarray(kept_rows, dtype=int)

    result = usable.iloc[kept_indices].copy()
    result = result.reset_index(drop=True)

    result["final_score"] = (
        models["rfr"]["model"].predict(X)
    )

    feature_names = models["rfr"].get(
        "feature_names",
        doc_feature_names(models["gmm"]["k"]),
    )

    result["baseline_score"] = X[
        :, feature_names.index("mean_p")
    ]

    elapsed = time.perf_counter() - start

    coverage["inference_seconds"] = elapsed
    coverage["milliseconds_per_document"] = (
        elapsed / len(result) * 1000
    )

    result["word_count"] = (
        result["text"].str.split().str.len()
    )

    # 평가 전에 정한 고정 기준
    result["length_group"] = pd.cut(
        result["word_count"],
        bins=[0, 50, 200, np.inf],
        labels=[
            "short_1_50",
            "medium_51_200",
            "long_201_plus",
        ],
        include_lowest=True,
    )

    y_true = result["is_ai"].to_numpy(dtype=int)
    overall_rows = []
    length_rows = []

    for model_name, column in [
        ("final_random_forest", "final_score"),
        ("mean_sentence_baseline", "baseline_score"),
    ]:
        scores = result[column].to_numpy()

        if (
            not np.isfinite(scores).all()
            or ((scores < 0) | (scores > 1)).any()
        ):
            raise ValueError(
                f"{name}/{model_name}: 유효하지 않은 점수입니다."
            )

        row = {
            "dataset": name,
            "model": model_name,
            **calculate_metrics(
                y_true, scores, args.threshold
            ),
        }

        lower, upper = bootstrap_f1(
            y_true,
            scores,
            args.threshold,
            args.bootstrap,
            args.seed,
        )

        row["f1_ci_low"] = lower
        row["f1_ci_high"] = upper
        overall_rows.append(row)

        for length_group, group in result.groupby(
            "length_group", observed=True
        ):
            length_rows.append({
                "dataset": name,
                "model": model_name,
                "length_group": str(length_group),
                **calculate_metrics(
                    group["is_ai"],
                    group[column],
                    args.threshold,
                ),
            })

    result["prediction"] = (
        result["final_score"] > args.threshold
    ).astype(int)

    result["error_type"] = np.select(
        [
            (result["is_ai"] == 0)
            & (result["prediction"] == 1),

            (result["is_ai"] == 1)
            & (result["prediction"] == 0),
        ],
        ["false_positive", "false_negative"],
        default="correct",
    )

    result.to_csv(
        output_dir / f"{name}_predictions.csv",
        index=False,
    )

    save_error_examples(
        result, name, output_dir, args.seed
    )
    save_plots(result, name, output_dir)

    return overall_rows, length_rows, coverage


# ---------------------------------------------------------
# 7. 보고서 초안
# ---------------------------------------------------------

def save_report(metrics, coverage, args, output_dir):
    report = [
        "# Model Evaluation",
        "",
        "Human=0; AI=1.",
        f"Prediction rule: score > {args.threshold}.",
        f"Random seed: {args.seed}.",
        f"Maximum documents: {args.max_docs or 'all'}.",
        "",
        "## Measured results",
        "",
        "```text",
        metrics.round(4).to_string(index=False),
        "```",
        "",
        "## Coverage and runtime",
        "",
        "```text",
        coverage.round(4).to_string(index=False),
        "```",
        "",
        "## Interpretation to complete",
        "",
        "- Compare the final model with the sentence-mean baseline.",
        "- Explain differences between test and external holdout.",
        "- Discuss length sensitivity using length_metrics.csv.",
        "- Include sample counts and class balance.",
        "- Read error_review.csv and fill manual_observation.",
        "- Explain patterns supported by actual error examples.",
        "",
        "## Limitations",
        "",
        "- Sentence labels are inherited from document labels.",
        "- Documents without a scorable sentence are excluded; "
        "their coverage is reported separately.",
        "- Duplicate stripped texts are removed within each "
        "evaluation dataset; conflicting labels cause an error.",
        "- Training overlap must be checked against the original "
        "training and validation files.",
        "- A 2026 filename does not prove a chronological holdout. "
        "Confirm collection dates and generation methods.",
        "- External domain and length shifts may be confounded.",
        "- F1 intervals use class-stratified document bootstrap "
        "and are conditional on the observed class mixture.",
        "- Random Forest scores are not independently calibrated.",
        "- Tree prediction percentiles are not statistical "
        "confidence intervals.",
        "- Runtime includes shared feature extraction and both "
        "predictions, not separate model speed measurements.",
        "- No retraining or test-set threshold tuning is performed.",
        "- Undefined precision/recall denominators are set to 0. "
        "ROC-AUC/AP are blank for single-class groups.",
    ]

    (output_dir / "evaluation_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# 8. 실행
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test",
        default="data/splits/test.parquet",
    )
    parser.add_argument(
        "--holdout",
        default="data/splits/holdout_2026.parquet",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="0 evaluates all documents",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        not 0 <= args.threshold <= 1
        or args.max_docs < 0
        or args.bootstrap < 0
    ):
        parser.error(
            "threshold는 0~1, max-docs와 bootstrap은 "
            "0 이상이어야 합니다."
        )

    datasets = [(Path(args.test), "test")]

    if not args.skip_holdout:
        datasets.append(
            (Path(args.holdout), "holdout_2026")
        )

    required_files = [path for path, _ in datasets]

    required_files += [
        Path(args.model_dir) / f"{name}.joblib"
        for name in ["classifier", "gmm", "regressor"]
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        parser.error(
            "다음 파일이 없습니다: "
            + ", ".join(missing)
            + "\n팀이 사용한 원래 분할 데이터가 필요합니다."
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_models(args.model_dir)

    metric_rows = []
    length_rows = []
    coverage_rows = []

    for path, name in datasets:
        print(f"Evaluating {name}...", flush=True)

        metrics, lengths, coverage = evaluate_dataset(
            path, name, models, args, output_dir
        )

        metric_rows.extend(metrics)
        length_rows.extend(lengths)
        coverage_rows.append(coverage)

    metrics_table = pd.DataFrame(metric_rows)
    length_table = pd.DataFrame(length_rows)
    coverage_table = pd.DataFrame(coverage_rows)

    metrics_table.to_csv(
        output_dir / "metrics_comparison.csv",
        index=False,
    )
    length_table.to_csv(
        output_dir / "length_metrics.csv",
        index=False,
    )
    coverage_table.to_csv(
        output_dir / "coverage.csv",
        index=False,
    )

    (output_dir / "settings.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    if not metrics_table.empty:
        chart_data = metrics_table.set_index(
            ["dataset", "model"]
        )[["accuracy", "precision", "recall", "f1"]]

        fig, ax = plt.subplots(figsize=(9, 5))
        chart_data.plot.bar(ax=ax)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Model performance comparison")
        ax.tick_params(axis="x", labelrotation=25)

        fig.tight_layout()
        fig.savefig(
            output_dir / "metrics_comparison.png",
            dpi=180,
        )
        plt.close(fig)

    save_report(
        metrics_table,
        coverage_table,
        args,
        output_dir,
    )

    print("\n=== Evaluation results ===")
    print(metrics_table.round(4).to_string(index=False))
    print(f"\n결과 저장 위치: {output_dir.resolve()}")


if __name__ == "__main__":
    main()