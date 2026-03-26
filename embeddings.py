"""ONNX-based sentence embeddings using all-MiniLM-L6-v2.

Uses ONNX Runtime for inference and the HuggingFace tokenizer directly,
bypassing ChromaDB (incompatible with Python 3.14) and sentence-transformers.
Produces 384-dimensional L2-normalized embeddings.
"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from pathlib import Path

import config


class Embedder:
    """Singleton-like embedding wrapper around ONNX all-MiniLM-L6-v2."""

    def __init__(self) -> None:
        model_dir = self._resolve_model_dir()
        tokenizer_path = str(model_dir / "tokenizer.json")
        model_path = str(model_dir / "onnx" / "model.onnx")

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=config.EMBEDDING_MAX_LENGTH)
        self._tokenizer.enable_padding(
            pad_id=0,
            pad_token="[PAD]",
            length=config.EMBEDDING_MAX_LENGTH,
        )

        self._session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

    def _resolve_model_dir(self) -> Path:
        """Find the cached model directory, downloading if needed."""
        cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        model_base = cache_base / "models--sentence-transformers--all-MiniLM-L6-v2"

        if model_base.exists():
            snapshots = model_base / "snapshots"
            if snapshots.exists():
                for snapshot in snapshots.iterdir():
                    onnx_path = snapshot / "onnx" / "model.onnx"
                    tokenizer_path = snapshot / "tokenizer.json"
                    if onnx_path.exists() and tokenizer_path.exists():
                        return snapshot

        # Fallback: download the required files
        tokenizer_path = hf_hub_download(
            config.EMBEDDING_MODEL_NAME, "tokenizer.json"
        )
        hf_hub_download(config.EMBEDDING_MODEL_NAME, "onnx/model.onnx")

        return Path(tokenizer_path).parent

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into 384-dim normalized vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), 384), L2-normalized.
        """
        if not texts:
            return np.zeros((0, config.EMBEDDING_DIM), dtype=np.float32)

        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array(
            [e.attention_mask for e in encodings], dtype=np.int64
        )
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling over token embeddings (masked)
        token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        embeddings = summed / counts

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns a 1D array of shape (384,)."""
        return self.embed([text])[0]

    def embed_batched(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed texts in batches to manage memory.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts per batch. Defaults to config.EMBED_BATCH_SIZE.

        Returns:
            numpy array of shape (len(texts), 384), L2-normalized.
        """
        if batch_size is None:
            batch_size = config.EMBED_BATCH_SIZE

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.append(self.embed(batch))

        return np.vstack(all_embeddings)
