"""Text preprocessing pipeline for fake news detection.

Single source of truth for all text cleaning operations used in
training, inference, and evaluation. Guarantees identical preprocessing
at all stages.

The pipeline is:
    1. Stringify + lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove special characters (keep only letters + spaces)
    5. Normalize whitespace

Design decisions:
    - No stemming/lemmatization: TF-IDF with ngrams handles morphology adequately
      and stemming adds a heavy dependency (spaCy/NLTK) with marginal F1 gain
      on this dataset.
    - Frozen config: prevents accidental mutation after pipeline is serialized.
    - Batch method: enables pandas .apply() replacement with vectorized processing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CleanerConfig:
    """Immutable cleaning configuration.

    Frozen dataclass ensures config cannot be mutated after model
    serialization, preventing train/serve skew.
    """
    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_special_chars: bool = True
    min_token_length: int = 1

    def fingerprint(self) -> str:
        """Deterministic hash of config for cache invalidation / audit."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class TextCleaner:
    """Deterministic text cleaning pipeline.

    Thread-safe, stateless (all state in config), serializable.

    Example:
        cleaner = TextCleaner()
        cleaned = cleaner.clean("Breaking!! Visit http://fake.com now")
        # => "breaking visit now"
    """

    # Pre-compiled regex patterns — compiled once at class level
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    _HTML_RE = re.compile(r"<[^>]+>")
    _SPECIAL_RE = re.compile(r"[^a-zA-Z\s]")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, config: CleanerConfig | None = None) -> None:
        self.config = config or CleanerConfig()

    def clean(self, text: str) -> str:
        """Apply the full cleaning pipeline to a single text.

        Args:
            text: Raw input text (handles None, NaN, numeric via str()).

        Returns:
            Cleaned text string. May be empty if input is all special chars.
        """
        text = str(text)

        if self.config.lowercase:
            text = text.lower()
        if self.config.remove_urls:
            text = self._URL_RE.sub("", text)
        if self.config.remove_html:
            text = self._HTML_RE.sub("", text)
        if self.config.remove_special_chars:
            text = self._SPECIAL_RE.sub("", text)

        text = self._WHITESPACE_RE.sub(" ", text).strip()
        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean a list of texts. Preserves order and length."""
        return [self.clean(t) for t in texts]

    def is_valid(self, text: str) -> bool:
        """Check if cleaned text meets minimum quality threshold."""
        cleaned = self.clean(text)
        tokens = cleaned.split()
        return len(tokens) >= self.config.min_token_length

    def __repr__(self) -> str:
        return f"TextCleaner(config={self.config}, fingerprint={self.config.fingerprint()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextCleaner):
            return NotImplemented
        return self.config == other.config