"""Lazy-loading ONNX text-embedding service using Alibaba-NLP/gte-modernbert-base.

Produces 768-dimensional L2-normalized sentence embeddings via onnxruntime (CPU).
Loading is lazy (first embed() call). Any failure sets _failed=True so that
embed()/embed_batch() return None and the memory system falls back to FTS-only
keyword search.
"""

import sys


_MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
_ONNX_FILENAME = "onnx/model.onnx"
_MODEL_MAX_TOKENS = 8192  # ModernBERT positional cap
EMBED_DIM = 768


# ---------------------------------------------------------------------------
# Provider-fallback helpers
# ---------------------------------------------------------------------------

METAL_TEXTURE_LIMIT = 16384
CPU_PROVIDER = "CPUExecutionProvider"
COREML_PROVIDER = "CoreMLExecutionProvider"


def _model_fits_coreml(model_path, limit=METAL_TEXTURE_LIMIT):
    """Return ``True`` if no ONNX initializer exceeds the 2-D texture ceiling."""
    import onnx

    m = onnx.load(str(model_path), load_external_data=False)
    return not any(d > limit for init in m.graph.initializer for d in init.dims)


def _choose_providers(model_path=None):
    """Pick execution providers, dropping CoreML when the model is too wide."""
    import onnxruntime as ort

    providers = list(ort.get_available_providers())
    if model_path is not None and COREML_PROVIDER in providers:
        if not _model_fits_coreml(model_path):
            providers = [p for p in providers if p != COREML_PROVIDER]
    if CPU_PROVIDER not in providers:
        providers.append(CPU_PROVIDER)
    return providers


def _build_session(model_path, providers):
    """Create an ``ort.InferenceSession`` with single-CPU fallback."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    try:
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    except Exception:
        if providers == [CPU_PROVIDER]:
            raise
        return ort.InferenceSession(str(model_path), sess_options=opts, providers=[CPU_PROVIDER])


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings over the sequence dimension."""
    import numpy as np

    mask = attention_mask[..., np.newaxis].astype(np.float32)
    sum_emb = (last_hidden_state * mask).sum(axis=1)
    sum_mask = mask.sum(axis=1).clip(min=1e-9)
    return sum_emb / sum_mask


def _l2_normalize(embeddings):
    """L2-normalise embeddings along the last axis, in-place-ish."""
    import numpy as np

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
    return embeddings / norms


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Thin wrapper around an ONNX ModernBERT embedding model."""

    __slots__ = (
        "_model_path",
        "_model_dir",
        "_loaded",
        "_failed",
        "_lock",
        "_session",
        "_tokenizer",
        "_input_names",
        "_output_names",
    )

    def __init__(self, model_path=None, model_dir=None):
        self._model_path = model_path
        self._model_dir = model_dir
        self._loaded = False
        self._failed = False
        import threading

        self._lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._input_names = []
        self._output_names = []

    def _ensure_loaded(self):
        """Resolve and load the model on first call (thread-safe).  Returns bool."""
        import os
        from pathlib import Path

        import numpy as np
        from transformers import AutoTokenizer

        if self._loaded:
            return True

        with self._lock:
            if self._loaded:
                return True
            try:
                # Resolve model path.
                model_dir = None
                onnx_path = None

                if self._model_path and os.path.exists(self._model_path):
                    onnx_path = Path(self._model_path)
                else:
                    model_dir = (
                        self._model_dir
                        or os.path.expanduser("~/.cache/coding_agent/models/gte-modernbert-base")
                    )
                    onnx_path = Path(model_dir) / "onnx/model.onnx"

                if not onnx_path.exists():
                    from huggingface_hub import hf_hub_download
                    import sys as _sys

                    _sys.stderr.write(
                        f"[memory] Downloading gte-modernbert-base ONNX to {model_dir or os.path.dirname(onnx_path)} ...\n"
                    )
                    # Ensure intermediate directories exist for the local-dir approach.
                    if model_dir is None:
                        model_dir = str(onnx_path.parent)
                    Path(model_dir).mkdir(parents=True, exist_ok=True)
                    hf_hub_download(
                        repo_id=_MODEL_ID,
                        filename=_ONNX_FILENAME,
                        local_dir=model_dir,
                    )

                # Tokenizer -- try local first, fall back to network.
                from_pretrained = AutoTokenizer.from_pretrained
                tokenizer = None
                try:
                    tokenizer = from_pretrained(_MODEL_ID, local_files_only=True)
                except Exception:
                    tokenizer = from_pretrained(_MODEL_ID)

                # Providers + session.
                providers = _choose_providers(str(onnx_path))
                session = _build_session(onnx_path, providers)

                input_names = [inp.name for inp in session.get_inputs()]
                output_names = [out.name for out in session.get_outputs()]

                self._session = session
                self._tokenizer = tokenizer
                self._input_names = input_names
                self._output_names = output_names
                self._loaded = True
                return True

            except Exception as exc:
                self._failed = True
                print(
                    f"[memory] embeddings unavailable, falling back to FTS-only keyword recall: {exc}",
                    file=sys.stderr,
                )
                return False

    @property
    def available(self):
        """Whether embeddings are usable."""
        # Optimistic: return True even before any load attempt.
        return not self._failed

    def embed(self, text) -> list | None:
        """Return a 768-float list for *text*, or ``None`` on failure."""
        import numpy as np

        if not self._ensure_loaded():
            return None

        tokenizer = self._tokenizer
        session = self._session

        inputs = tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=_MODEL_MAX_TOKENS,
        )

        # Prepare model inputs.
        feed_dict = {
            "input_ids": np.asarray(inputs["input_ids"]),
            "attention_mask": np.asarray(inputs["attention_mask"]),
        }

        if "token_type_ids" in self._input_names:
            feed_dict["token_type_ids"] = np.zeros_like(feed_dict["input_ids"])

        outputs = session.run(self._output_names, feed_dict)

        # Find output tensor.
        embedding = None
        for out_name, out_data in zip(self._output_names, outputs):
            if out_name == "sentence_embedding":
                embedding = out_data
                break

        if embedding is None:
            # Use mean pooling when explicit output isn't named sentence_embedding.
            if "last_hidden_state" in self._output_names:
                last_hidden_state = outputs[self._output_names.index("last_hidden_state")]
            else:
                last_hidden_state = outputs[0]
            embedding = _mean_pool(last_hidden_state, feed_dict["attention_mask"])

        embedding = _l2_normalize(embedding).astype(np.float32)
        return embedding.flatten().tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Return *n* x 768-float lists for each string in *texts*, or ``None``."""
        import numpy as np

        if not self._ensure_loaded():
            return None

        tokens_list = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i : i + 32]
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=_MODEL_MAX_TOKENS,
            )
            tokens_list.append(inputs)

        results: list[list[float]] = []

        for inputs in tokens_list:
            feed_dict = {
                "input_ids": np.asarray(inputs["input_ids"]),
                "attention_mask": np.asarray(inputs["attention_mask"]),
            }

            if "token_type_ids" in self._input_names:
                feed_dict["token_type_ids"] = np.zeros_like(feed_dict["input_ids"])

            outputs = self._session.run(self._output_names, feed_dict)

            embedding = None
            for out_name, out_data in zip(self._output_names, outputs):
                if out_name == "sentence_embedding":
                    embedding = out_data
                    break

            if embedding is None:
                if "last_hidden_state" in self._output_names:
                    last_hidden_state = outputs[self._output_names.index("last_hidden_state")]
                else:
                    last_hidden_state = outputs[0]
                embedding = _mean_pool(last_hidden_state, feed_dict["attention_mask"])

            embedding = _l2_normalize(embedding).astype(np.float32)
            for row in range(embedding.shape[0]):
                results.append(embedding[row].flatten().tolist())

        return results
