"""Classical ML training pipeline with MLflow experiment tracking.

Evaluates Logistic Regression, Linear SVM, and Multinomial NB across a
grid of TF-IDF configurations (ngrams × class_weights) using stratified
K-fold cross-validation.

All runs are logged to MLflow with:
    - Parameters (model, ngram, class_weight, max_features, etc.)
    - Metrics (mean_f1, std_f1, per-fold scores)
    - Artifacts (serialized model + vectorizer)
    - Tags (dataset fingerprint, cleaner config hash)

Usage:
    # From CLI
    python -m src.training.train_classical

    # Programmatic
    trainer = ClassicalTrainer()
    trainer.load_data()
    results = trainer.run_evaluation()
    tuned = trainer.tune_best_model()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from configs.settings import get_settings
from src.data_pipeline.loader import DataLoader, DataResult
from src.utils.logger import get_logger
from src.utils.versioning import get_next_version, save_model_metadata

logger = get_logger(__name__)


# ── Default model zoo ────────────────────────────────────────────────

def get_default_models() -> dict[str, BaseEstimator]:
    """Return fresh model instances (never reuse fitted estimators)."""
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "LinearSVC": LinearSVC(max_iter=5000, dual="auto"),
        "MultinomialNB": MultinomialNB(),
    }


NGRAM_OPTIONS: list[tuple[int, int]] = [(1, 1), (1, 2)]
CLASS_WEIGHT_OPTIONS: list[str | None] = [None, "balanced"]


# ── Result container ─────────────────────────────────────────────────

@dataclass
class RunResult:
    """Structured output for a single model × config evaluation."""
    model_name: str
    ngram_range: tuple[int, int]
    class_weight: str | None
    mean_f1: float
    std_f1: float
    fold_scores: list[float]
    precision: float
    recall: float
    training_time_s: float
    version: str

    @property
    def summary(self) -> str:
        return (
            f"{self.model_name} | ngram={self.ngram_range} | cw={self.class_weight} | "
            f"F1={self.mean_f1:.4f}±{self.std_f1:.4f} | P={self.precision:.4f} R={self.recall:.4f}"
        )


# ── Trainer ──────────────────────────────────────────────────────────

class ClassicalTrainer:
    """Orchestrates cross-validated evaluation + MLflow logging."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.data_result: DataResult | None = None

    @property
    def X(self) -> pd.Series:
        if self.data_result is None:
            raise RuntimeError("Call load_data() first.")
        return self.data_result.X

    @property
    def y(self) -> pd.Series:
        if self.data_result is None:
            raise RuntimeError("Call load_data() first.")
        return self.data_result.y

    def load_data(self, file_path: str | Path | None = None) -> DataResult:
        """Load and preprocess dataset."""
        loader = DataLoader(file_path)
        self.data_result = loader.prepare()
        return self.data_result

    def _build_tfidf(self, ngram: tuple[int, int]) -> TfidfVectorizer:
        """Construct TF-IDF vectorizer from settings."""
        s = self.settings
        return TfidfVectorizer(
            max_features=s.tfidf_max_features,
            ngram_range=ngram,
            stop_words="english",
            min_df=s.tfidf_min_df,
            max_df=s.tfidf_max_df,
            sublinear_tf=s.tfidf_sublinear_tf,
        )

    def run_evaluation(
        self,
        models: dict[str, BaseEstimator] | None = None,
        ngrams: list[tuple[int, int]] | None = None,
        weights: list[str | None] | None = None,
    ) -> pd.DataFrame:
        """Cross-validated grid over models × ngrams × class_weights.

        Returns:
            DataFrame sorted by mean_f1 descending with all run metadata.
        """
        models = models or get_default_models()
        ngrams = ngrams or NGRAM_OPTIONS
        weights = weights or CLASS_WEIGHT_OPTIONS

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment)

        skf = StratifiedKFold(
            n_splits=self.settings.cv_folds,
            shuffle=True,
            random_state=self.settings.random_state,
        )

        results: list[RunResult] = []

        total_configs = sum(
            1 for _ in models for _ in ngrams for w in weights
            if w != "balanced" or hasattr(list(models.values())[0], "class_weight")
        )
        logger.info("Starting evaluation: %d configurations", total_configs)

        for model_name, model_obj in models.items():
            for ngram in ngrams:
                for weight in weights:
                    if weight == "balanced" and not hasattr(model_obj, "class_weight"):
                        continue

                    result = self._evaluate_config(
                        model_name, model_obj, ngram, weight, skf
                    )
                    results.append(result)
                    logger.info(result.summary)

        # Build results DataFrame
        df = pd.DataFrame([
            {
                "model_name": r.model_name,
                "ngram_range": str(r.ngram_range),
                "class_weight": str(r.class_weight),
                "mean_f1": round(r.mean_f1, 6),
                "std_f1": round(r.std_f1, 6),
                "precision": round(r.precision, 6),
                "recall": round(r.recall, 6),
                "training_time_s": round(r.training_time_s, 2),
                "version": r.version,
                "fold_scores": r.fold_scores,
            }
            for r in results
        ]).sort_values("mean_f1", ascending=False).reset_index(drop=True)

        # Save results
        out = self.settings.experiment_dir / "baseline_results.csv"
        df.to_csv(out, index=False)
        logger.info("Results saved to %s", out)

        return df

    def _evaluate_config(
        self,
        name: str,
        model_obj: BaseEstimator,
        ngram: tuple[int, int],
        weight: str | None,
        skf: StratifiedKFold,
    ) -> RunResult:
        """Run K-fold CV for a single configuration and log to MLflow."""
        run_tag = f"{name}_ngram{ngram}_cw{weight}"
        logger.info("Evaluating: %s", run_tag)

        fold_f1: list[float] = []
        fold_precision: list[float] = []
        fold_recall: list[float] = []

        start_time = time.perf_counter()

        with mlflow.start_run(run_name=run_tag):
            mlflow.log_params({
                "model_name": name,
                "ngram_range": str(ngram),
                "class_weight": str(weight),
                "max_features": self.settings.tfidf_max_features,
                "min_df": self.settings.tfidf_min_df,
                "max_df": self.settings.tfidf_max_df,
                "sublinear_tf": self.settings.tfidf_sublinear_tf,
                "cv_folds": self.settings.cv_folds,
                "n_samples": len(self.X),
            })

            # Log data audit metadata
            if self.data_result:
                mlflow.log_params({
                    f"data_{k}": str(v)
                    for k, v in self.data_result.metadata.items()
                    if k in ("raw_rows", "final_rows", "cleaner_fingerprint")
                })

            last_model = None
            last_tfidf = None

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(self.X, self.y)):
                X_tr, X_te = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_tr, y_te = self.y.iloc[train_idx], self.y.iloc[test_idx]

                tfidf = self._build_tfidf(ngram)
                X_tr_vec = tfidf.fit_transform(X_tr)
                X_te_vec = tfidf.transform(X_te)

                model = clone(model_obj)
                if hasattr(model, "class_weight"):
                    model.set_params(class_weight=weight)

                model.fit(X_tr_vec, y_tr)
                y_pred = model.predict(X_te_vec)

                f1 = f1_score(y_te, y_pred, average="binary")
                p, r, _, _ = precision_recall_fscore_support(
                    y_te, y_pred, average="binary", zero_division=0
                )

                fold_f1.append(f1)
                fold_precision.append(p)
                fold_recall.append(r)

                mlflow.log_metric(f"fold_{fold_idx}_f1", f1, step=fold_idx)

                last_model = model
                last_tfidf = tfidf

            elapsed = time.perf_counter() - start_time

            mean_f1 = float(np.mean(fold_f1))
            std_f1 = float(np.std(fold_f1))
            mean_p = float(np.mean(fold_precision))
            mean_r = float(np.mean(fold_recall))

            mlflow.log_metrics({
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "training_time_s": elapsed,
            })

            # Save versioned artifacts
            version = get_next_version()
            model_dir = self.settings.model_dir

            model_path = model_dir / f"{version}.pkl"
            vec_path = model_dir / f"{version}_vectorizer.pkl"

            joblib.dump(last_model, model_path)
            joblib.dump(last_tfidf, vec_path)

            # Save metadata sidecar
            save_model_metadata(
                version=version,
                metrics={"mean_f1": mean_f1, "std_f1": std_f1, "precision": mean_p, "recall": mean_r},
                config={"model": name, "ngram": str(ngram), "class_weight": str(weight)},
            )

            mlflow.sklearn.log_model(last_model, "model")
            mlflow.log_param("saved_version", version)

        return RunResult(
            model_name=name,
            ngram_range=ngram,
            class_weight=weight,
            mean_f1=mean_f1,
            std_f1=std_f1,
            fold_scores=[round(s, 6) for s in fold_f1],
            precision=mean_p,
            recall=mean_r,
            training_time_s=elapsed,
            version=version,
        )

    def tune_best_model(self) -> Pipeline:
        """GridSearchCV on Logistic Regression pipeline.

        Uses settings for param grid. Returns fitted best pipeline.
        """
        s = self.settings
        logger.info("Starting hyperparameter tuning...")

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                stop_words="english",
                sublinear_tf=s.tfidf_sublinear_tf,
                min_df=s.tfidf_min_df,
                max_df=s.tfidf_max_df,
            )),
            ("model", LogisticRegression(max_iter=1000)),
        ])

        param_grid = {
            "tfidf__max_features": s.tune_max_features,
            "tfidf__ngram_range": s.tune_ngram_ranges,
            "model__C": s.tune_C_values,
            "model__class_weight": s.tune_class_weights,
        }

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=s.cv_folds,
            scoring="f1",
            n_jobs=s.n_jobs,
            verbose=1,
            return_train_score=True,
        )

        start = time.perf_counter()
        grid.fit(self.X, self.y)
        elapsed = time.perf_counter() - start

        logger.info("Tuning complete in %.1fs", elapsed)
        logger.info("Best params: %s", grid.best_params_)
        logger.info("Best F1: %.4f", grid.best_score_)

        # Check for overfitting
        best_idx = grid.best_index_
        train_score = grid.cv_results_["mean_train_score"][best_idx]
        test_score = grid.cv_results_["mean_test_score"][best_idx]
        gap = train_score - test_score
        if gap > 0.05:
            logger.warning(
                "Possible overfitting: train_f1=%.4f, test_f1=%.4f, gap=%.4f",
                train_score, test_score, gap,
            )

        # Save
        out = self.settings.model_dir / "best_tuned_pipeline.pkl"
        joblib.dump(grid.best_estimator_, out)
        logger.info("Tuned pipeline saved to %s", out)

        # Log to MLflow
        with mlflow.start_run(run_name="hyperparameter_tuning"):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "best_f1": grid.best_score_,
                "train_f1": train_score,
                "test_f1": test_score,
                "overfit_gap": gap,
                "tuning_time_s": elapsed,
            })
            mlflow.sklearn.log_model(grid.best_estimator_, "tuned_pipeline")

        return grid.best_estimator_


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ClassicalTrainer()
    data = trainer.load_data()

    print(f"\nDataset: {data.n_samples} samples")
    print(f"Class distribution: {data.class_distribution}")
    print(f"Audit: {data.metadata}\n")

    results = trainer.run_evaluation()
    print("\n" + results.drop(columns=["fold_scores"]).to_string(index=False))

    best = results.iloc[0]
    print(f"\nBest: {best['model_name']} (F1={best['mean_f1']:.4f})")

    print("\nStarting hyperparameter tuning...")
    tuned = trainer.tune_best_model()
    print("Done.")