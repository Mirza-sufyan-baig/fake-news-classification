"""BERT-based fake news classifier using HuggingFace Transformers.

Production features:
    - Mixed precision training (fp16) when GPU available
    - Learning rate warmup schedule
    - Weight decay regularization
    - Early stopping via load_best_model_at_end
    - Checkpoint resumption
    - Structured metrics logging

Usage:
    python -m src.models.bert_classifier
    # or
    FND_BERT_EPOCHS=3 FND_BERT_BATCH_SIZE=16 python -m src.models.bert_classifier
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from configs.settings import get_settings
from src.data_pipeline.loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FakeNewsDataset(Dataset):
    """PyTorch Dataset for tokenized news articles.

    Tokenization happens lazily in __getitem__ to avoid holding
    the entire tokenized dataset in memory.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_len: int = 256,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(pred: Any) -> dict[str, float]:
    """Compute metrics for HuggingFace Trainer callback."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


class BERTTrainer:
    """End-to-end BERT fine-tuning pipeline with production hardening."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def train(self, file_path: str | Path | None = None) -> dict[str, float]:
        """Fine-tune BERT on the fake news dataset.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Lazy imports — don't load transformers unless BERT training is requested
        from transformers import (
            BertForSequenceClassification,
            BertTokenizer,
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
        )

        s = self.settings

        # Load data
        loader = DataLoader(file_path)
        data = loader.prepare()
        texts = data.X.tolist()
        labels = data.y.tolist()

        logger.info("BERT training on %d samples", len(texts))

        # Stratified split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=s.test_size,
            random_state=s.random_state,
            stratify=labels,
        )

        logger.info("Train: %d | Val: %d", len(train_texts), len(val_texts))

        # Tokenizer & datasets
        tokenizer = BertTokenizer.from_pretrained(s.bert_model_name)
        train_ds = FakeNewsDataset(train_texts, train_labels, tokenizer, s.bert_max_len)
        val_ds = FakeNewsDataset(val_texts, val_labels, tokenizer, s.bert_max_len)

        # Model
        model = BertForSequenceClassification.from_pretrained(
            s.bert_model_name, num_labels=2
        )

        # Training args with production settings
        output_dir = str(s.model_dir / "bert_output")
        use_fp16 = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=s.bert_lr,
            per_device_train_batch_size=s.bert_batch_size,
            per_device_eval_batch_size=s.bert_batch_size * 2,  # eval can use larger batch
            num_train_epochs=s.bert_epochs,
            warmup_ratio=s.bert_warmup_ratio,
            weight_decay=s.bert_weight_decay,
            fp16=use_fp16,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            logging_dir=str(s.log_dir / "bert"),
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=3,  # keep only 3 best checkpoints
            report_to="none",  # disable wandb/tensorboard unless configured
            dataloader_num_workers=2 if not os.name == "nt" else 0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Resume from checkpoint if available
        resume = None
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                resume = output_dir
                logger.info("Resuming from checkpoint in %s", output_dir)

        trainer.train(resume_from_checkpoint=resume)

        # Evaluate
        metrics = trainer.evaluate()
        logger.info("BERT eval metrics: %s", metrics)

        # Save final model
        save_path = str(s.model_dir / "bert_fakenews_final")
        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info("BERT model saved to %s", save_path)

        return metrics


if __name__ == "__main__":
    metrics = BERTTrainer().train()
    print(f"\nFinal metrics: {metrics}")