"""Data loading, validation, quality checks, and preprocessing.

Responsibilities:
    1. Load CSV with schema validation
    2. Run data quality checks (null ratios, class balance, duplicates)
    3. Apply text cleaning pipeline
    4. Encode labels
    5. Return clean (X, y) pair with audit metadata

Design:
    - DataLoader is stateless — every call to prepare() is idempotent
    - Quality issues are logged as warnings, not silently dropped
    - Returns DataResult with metadata for experiment tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from configs.settings import get_settings
from src.core.exceptions import DataNotFoundError, DataQualityError, DataSchemaError
from src.features.cleaner import TextCleaner
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataResult:
    """Container for preprocessed data with audit metadata."""
    X: pd.Series
    y: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def class_distribution(self) -> dict[int, int]:
        return self.y.value_counts().to_dict()


class DataLoader:
    """Load, validate, and preprocess the fake news dataset.

    Example:
        loader = DataLoader()
        result = loader.prepare()
        print(result.n_samples, result.class_distribution)
    """

    def __init__(self, file_path: Path | str | None = None) -> None:
        settings = get_settings()
        self.file_path = Path(file_path) if file_path else settings.dataset_path
        self.cleaner = TextCleaner()
        self._settings = settings

    def load(self) -> pd.DataFrame:
        """Load raw CSV with schema validation."""
        logger.info("Loading dataset from %s", self.file_path)

        if not self.file_path.exists():
            raise DataNotFoundError(
                f"Dataset not found: {self.file_path}",
                context={"path": str(self.file_path)},
            )

        df = pd.read_csv(self.file_path)
        logger.info("Raw shape: %s", df.shape)

        # Schema validation
        required = {self._settings.text_column, self._settings.label_column}
        missing = required - set(df.columns)
        if missing:
            raise DataSchemaError(
                f"Missing required columns: {missing}",
                context={"required": list(required), "found": list(df.columns)},
            )

        return df

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run data quality checks. Returns audit dict."""
        s = self._settings
        audit: dict[str, Any] = {
            "raw_rows": len(df),
            "columns": list(df.columns),
        }

        # Null ratio check
        text_nulls = df[s.text_column].isna().mean()
        label_nulls = df[s.label_column].isna().mean()
        audit["text_null_ratio"] = round(text_nulls, 4)
        audit["label_null_ratio"] = round(label_nulls, 4)

        if text_nulls > s.max_null_ratio:
            logger.warning(
                "High null ratio in '%s': %.1f%% (threshold: %.1f%%)",
                s.text_column, text_nulls * 100, s.max_null_ratio * 100,
            )

        # Duplicate check
        n_dup_rows = df.duplicated().sum()
        n_dup_text = df.duplicated(subset=[s.text_column]).sum()
        audit["duplicate_rows"] = int(n_dup_rows)
        audit["duplicate_texts"] = int(n_dup_text)

        if n_dup_text > 0:
            logger.warning(
                "%d duplicate texts found (%.1f%% of dataset)",
                n_dup_text, n_dup_text / len(df) * 100,
            )

        # Label distribution
        label_counts = df[s.label_column].value_counts()
        audit["label_distribution"] = label_counts.to_dict()

        unknown_labels = set(df[s.label_column].dropna().unique()) - set(s.label_map.keys())
        if unknown_labels:
            logger.warning("Unknown labels found: %s", unknown_labels)
            audit["unknown_labels"] = list(unknown_labels)

        # Class imbalance check
        if len(label_counts) >= 2:
            ratio = label_counts.min() / label_counts.max()
            audit["class_balance_ratio"] = round(ratio, 4)
            if ratio < 0.3:
                logger.warning(
                    "Severe class imbalance: minority/majority = %.2f", ratio
                )

        # Text length stats
        text_lengths = df[s.text_column].astype(str).str.split().str.len()
        audit["text_length_stats"] = {
            "mean": round(text_lengths.mean(), 1),
            "median": round(text_lengths.median(), 1),
            "min": int(text_lengths.min()),
            "max": int(text_lengths.max()),
            "std": round(text_lengths.std(), 1),
        }

        logger.info("Data validation complete: %s", audit)
        return audit

    def prepare(self, df: pd.DataFrame | None = None) -> DataResult:
        """Full pipeline: load → validate → clean → encode → return.

        Returns:
            DataResult containing (X, y) and audit metadata.
        """
        if df is None:
            df = self.load()

        s = self._settings

        # Validate
        audit = self.validate(df)

        # Drop nulls
        before = len(df)
        df = df.dropna(subset=[s.text_column, s.label_column]).reset_index(drop=True)
        dropped_nulls = before - len(df)
        if dropped_nulls > 0:
            logger.info("Dropped %d rows with missing values", dropped_nulls)

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=[s.text_column]).reset_index(drop=True)
        dropped_dups = before - len(df)
        if dropped_dups > 0:
            logger.info("Dropped %d duplicate texts", dropped_dups)

        # Filter unknown labels
        df = df[df[s.label_column].isin(s.label_map.keys())].reset_index(drop=True)

        # Clean text
        df["cleaned_text"] = df[s.text_column].apply(self.cleaner.clean)

        # Remove empty-after-cleaning rows
        mask = df["cleaned_text"].str.strip() != ""
        dropped_empty = (~mask).sum()
        df = df[mask].reset_index(drop=True)
        if dropped_empty > 0:
            logger.info("Dropped %d rows empty after cleaning", dropped_empty)

        # Encode labels
        df["label_encoded"] = df[s.label_column].map(s.label_map)

        X = df["cleaned_text"]
        y = df["label_encoded"]

        # Final audit
        audit.update({
            "dropped_nulls": dropped_nulls,
            "dropped_duplicates": dropped_dups,
            "dropped_empty": int(dropped_empty),
            "final_rows": len(X),
            "final_class_distribution": y.value_counts().to_dict(),
            "cleaner_fingerprint": self.cleaner.config.fingerprint(),
        })

        logger.info(
            "Preprocessing complete: %d → %d samples | Distribution: %s",
            audit["raw_rows"], len(X), y.value_counts().to_dict(),
        )

        return DataResult(X=X, y=y, metadata=audit)