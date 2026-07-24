import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


DEFAULT_BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def normalize_embedding_backend(backend: str, build_embeddings: bool = False) -> str:
    value = (backend or "lexical").strip().lower()
    if value in {"bge", "bge_transformers", "sentence_transformers"}:
        return "bge_transformers"
    if value == "auto" and build_embeddings:
        return "bge_transformers"
    return "lexical"


def resolve_embedding_model_path(model_path: Optional[str] = None) -> str:
    return (
        model_path
        or os.environ.get("HARNESS_G_EMBEDDING_MODEL_PATH")
        or os.environ.get("BGE_MODEL_PATH")
        or DEFAULT_BGE_MODEL
    )


def resolve_embedding_device(device: Optional[str] = None) -> str:
    if device:
        return device
    env_device = os.environ.get("HARNESS_G_EMBEDDING_DEVICE")
    if env_device:
        return env_device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TransformerEmbeddingModel:
    """Minimal BGE-compatible encoder using Transformers, without sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = str(model_name_or_path)
        self.device = resolve_embedding_device(device)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(self.model_name_or_path, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = int(getattr(self.model.config, "hidden_size", 0))

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        query_instruction: str = "",
        show_progress: bool = False,
    ) -> np.ndarray:
        import torch

        items = [str(text or "") for text in texts]
        if not items:
            return np.zeros((0, self.embedding_dim), dtype="float32")

        iterator = range(0, len(items), max(int(batch_size), 1))
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=(len(items) + max(int(batch_size), 1) - 1) // max(int(batch_size), 1), desc="BGE encode")
            except Exception:
                pass

        outputs: List[np.ndarray] = []
        with torch.inference_mode():
            for start in iterator:
                batch = items[start : start + max(int(batch_size), 1)]
                if query_instruction:
                    batch = [query_instruction + text for text in batch]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                model_output = self.model(**encoded)
                embeddings = model_output.last_hidden_state[:, 0]
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.detach().cpu().float().numpy())
        return np.concatenate(outputs, axis=0).astype("float32", copy=False)


def embedding_model_is_local(model_path: str) -> bool:
    path = Path(str(model_path))
    return path.exists() and path.is_dir()
