"""
Shared, lazy-loaded sentence-transformer embedder singleton.

Mirrors the model used by ToolRetriever (``wrapper.retriever.model_path``)
so belief embeddings and tool-corpus embeddings live in the same space —
avoiding encoder skew when computing ρ(b_g*, t*).

Configuration precedence (highest to lowest):
    1. Environment variable ``BELIEF_EMBEDDER_MODEL``
    2. ``configs/_base/embedder.yaml`` key ``pin`` or ``name``
    3. ``configs/model_pins.yaml`` key ``embedders.toolret``

Embedding cache:
    Encoded texts are memoised under ``FITTEXT_RUNTIME_ROOT`` so
    repeated calls with the same string are free (important when the same
    pseudo-tool text appears across generations in the same run).

Usage::

    emb = SharedEmbedder.instance()
    vecs = emb.encode(["pseudo-tool desc", "gold tool desc"])
    # vecs shape: (2, D), L2-normalised float32
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_PATH = _REPO_ROOT / "configs" / "_base" / "embedder.yaml"
_PIN_PATH = _REPO_ROOT / "configs" / "model_pins.yaml"


def _load_embedder_config() -> dict[str, Any]:
    """Load embedder config from YAML, returning {} if file absent.

    Returns:
        Dict with at least key ``pin`` or ``name`` if the file exists and is valid.
    """
    if not _CONFIG_PATH.exists():
        return {}
    try:
        import yaml  # type: ignore

        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.warning("Could not load embedder config from %s: %s", _CONFIG_PATH, exc)
        return {}


def _resolve_model_name() -> str:
    """Resolve the embedder model name from env → config → pin file.

    Returns:
        Model name string suitable for SentenceTransformer or HuggingFace AutoModel.
    """
    env_model = os.environ.get("BELIEF_EMBEDDER_MODEL", "").strip()
    if env_model:
        logger.info("SharedEmbedder: using model from env BELIEF_EMBEDDER_MODEL=%s", env_model)
        return env_model

    cfg = _load_embedder_config()
    cfg_model = str(cfg.get("name", "") or cfg.get("model", "")).strip()
    if cfg_model:
        logger.info("SharedEmbedder: using model from embedder.yaml: %s", cfg_model)
        return cfg_model

    cfg_pin = str(cfg.get("pin", "")).strip()
    try:
        import yaml  # type: ignore

        with open(_PIN_PATH, encoding="utf-8") as fh:
            pins = yaml.safe_load(fh) or {}
        if cfg_pin:
            node: Any = pins
            for part in cfg_pin.split("."):
                node = node[part]
            logger.info("SharedEmbedder: using model pin %s", cfg_pin)
            return str(node)
        logger.info("SharedEmbedder: using embedders.toolret from model_pins.yaml")
        return str(pins["embedders"]["toolret"])
    except Exception as exc:
        raise RuntimeError(f"Could not resolve SharedEmbedder model from {_PIN_PATH}") from exc


def _resolve_model_revision() -> str | None:
    """Resolve optional pinned revision (git SHA or tag) from config.

    Returns:
        Revision string, or None if not pinned.
    """
    env_rev = os.environ.get("BELIEF_EMBEDDER_REVISION", "").strip()
    if env_rev:
        return env_rev
    cfg = _load_embedder_config()
    return cfg.get("revision") or None


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    """Determine per-job embedding cache directory.

    Uses ``JOB_ID`` env var if set (SLURM / PBS), under FITTEXT_RUNTIME_ROOT.

    Returns:
        Path to cache directory (created on demand by the caller).
    """
    job_id = os.environ.get("JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"
    runtime_root = Path(os.environ.get("FITTEXT_RUNTIME_ROOT", str(_REPO_ROOT / "runs")))
    return runtime_root / "belief_embed_cache" / job_id


def _text_hash(text: str) -> str:
    """SHA-256 hex digest of a UTF-8 text string (truncated to 16 chars).

    Args:
        text: Input string.

    Returns:
        16-character hex string used as cache key.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SharedEmbedder
# ---------------------------------------------------------------------------

class SharedEmbedder:
    """Singleton, lazy-loaded sentence-transformer wrapper.

    Thread-safe; the underlying model is loaded exactly once per process.
    Embeddings are L2-normalised (unit norm) for cosine similarity via dot
    product, consistent with ToolRetriever's retrieval scoring.

    The real sentence-transformers library is imported only at first
    ``encode()`` call, so importing this module in tests that mock the
    embedder is free.

    Attributes:
        model_name: Resolved model name (read after first encode).
        model_revision: Pinned revision or None.
    """

    _instance: "SharedEmbedder | None" = None
    _instance_lock: Lock = Lock()

    def __init__(self) -> None:
        self._model_name: str = _resolve_model_name()
        self._model_revision: str | None = _resolve_model_revision()
        self._model: Any = None  # SentenceTransformer or AutoModel wrapper
        self._tokenizer: Any = None
        self._load_lock: Lock = Lock()
        self._cache: dict[str, np.ndarray] = {}  # text_hash → (D,) vector
        self._cache_dir: Path = _cache_dir()
        self._cache_loaded = False

    @classmethod
    def instance(cls) -> "SharedEmbedder":
        """Return the process-global singleton instance.

        Returns:
            SharedEmbedder singleton (created on first call).
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def model_name(self) -> str:
        """Resolved model name string."""
        return self._model_name

    @property
    def model_revision(self) -> str | None:
        """Pinned revision string, or None."""
        return self._model_revision

    # -- Lazy load ---------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded (thread-safe).

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            logger.info(
                "SharedEmbedder: loading %s (revision=%s)",
                self._model_name,
                self._model_revision,
            )
            if "simcse" in self._model_name.lower():
                self._load_simcse()
            else:
                self._load_sentence_transformer()
            logger.info("SharedEmbedder: model loaded.")
            self._load_disk_cache()

    def _load_sentence_transformer(self) -> None:
        """Load model via sentence-transformers."""
        from sentence_transformers import SentenceTransformer  # type: ignore

        kwargs: dict[str, Any] = {}
        if self._model_revision:
            kwargs["revision"] = self._model_revision
        self._model = SentenceTransformer(self._model_name, **kwargs)

    def _load_simcse(self) -> None:
        """Load sup-SimCSE model via HuggingFace transformers."""
        import torch
        from transformers import AutoModel, AutoTokenizer  # type: ignore

        kwargs: dict[str, Any] = {}
        if self._model_revision:
            kwargs["revision"] = self._model_revision
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, **kwargs)
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(self._model_name, **kwargs).to(device)
        self._simcse_device = device

    # -- Disk cache --------------------------------------------------------

    def _load_disk_cache(self) -> None:
        """Load per-job embedding cache from disk (if present).

        The cache is a pickle of ``{text_hash: np.ndarray}`` pairs.  Loading
        is best-effort — any error is logged and ignored.
        """
        cache_file = self._cache_dir / "embed_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as fh:
                    loaded: dict[str, np.ndarray] = pickle.load(fh)
                self._cache.update(loaded)
                logger.debug(
                    "SharedEmbedder: loaded %d cached embeddings from %s",
                    len(loaded),
                    cache_file,
                )
            except Exception as exc:
                logger.warning("SharedEmbedder: could not load disk cache: %s", exc)
        self._cache_loaded = True

    def _persist_disk_cache(self) -> None:
        """Persist in-memory cache to disk (best-effort)."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self._cache_dir / "embed_cache.pkl"
            with open(cache_file, "wb") as fh:
                pickle.dump(dict(self._cache), fh, protocol=4)
        except Exception as exc:
            logger.warning("SharedEmbedder: could not persist disk cache: %s", exc)

    # -- Encoding ----------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalised embedding vectors.

        Cache hits are returned without model inference.  Cache misses are
        batched into a single model call, then cached.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), D), dtype float32, L2-normalised
            so that cosine similarity == dot product.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_loaded()

        # Partition into cache hits and misses
        hashes = [_text_hash(t) for t in texts]
        miss_indices = [i for i, h in enumerate(hashes) if h not in self._cache]
        miss_texts = [texts[i] for i in miss_indices]

        if miss_texts:
            new_vecs = self._encode_batch(miss_texts)  # (M, D)
            for i, vec in zip(miss_indices, new_vecs):
                self._cache[hashes[i]] = vec
            # Persist after each batch (best-effort; keeps cache warm across runs)
            self._persist_disk_cache()

        # Assemble result in input order
        dim = next(iter(self._cache.values())).shape[0] if self._cache else 1
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for i, h in enumerate(hashes):
            result[i] = self._cache[h]
        return result

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Run model inference on a batch of texts (no caching).

        Args:
            texts: Non-empty list of strings to encode.

        Returns:
            np.ndarray shape (len(texts), D), L2-normalised, float32.
        """
        if "simcse" in self._model_name.lower():
            return self._encode_simcse(texts)
        else:
            return self._encode_st(texts)

    def _encode_st(self, texts: list[str]) -> np.ndarray:
        """Encode via SentenceTransformer with L2 normalisation.

        Args:
            texts: Strings to encode.

        Returns:
            L2-normalised float32 array.
        """
        from sentence_transformers import util as st_util  # type: ignore

        vecs = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    def _encode_simcse(self, texts: list[str]) -> np.ndarray:
        """Encode via sup-SimCSE (HuggingFace) with CLS pooling + L2 norm.

        Args:
            texts: Strings to encode.

        Returns:
            L2-normalised float32 array.
        """
        import torch

        all_vecs: list[np.ndarray] = []
        batch_size = 32
        device = self._simcse_device
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = self._model(**enc)
            # CLS token as sentence representation
            cls = out.last_hidden_state[:, 0, :]  # (B, D)
            cls_np = cls.cpu().float().numpy()
            # L2 normalise
            norms = np.linalg.norm(cls_np, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_vecs.append(cls_np / norms)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)
