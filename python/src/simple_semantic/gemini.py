"""``GeminiEmbedder`` — the one component that makes a network call.

In its own module, and ``httpx`` is the ``gemini`` extra rather than a core
dependency, so importing :mod:`simple_semantic` neither loads an HTTP client nor
requires one to be installed::

    pip install simple-semantic[gemini]
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
import numpy as np

from .embedder import normalize_row, normalize_rows

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"

#: Two model quirks that are silent when you get them wrong:
#:
#: gemini-embedding-001 pre-normalizes only its default 3072-dimension output;
#: anything smaller comes back unnormalized. gemini-embedding-002 aggregates a
#: multi-input request into one embedding unless each input is wrapped
#: individually. This class normalizes unconditionally and always wraps
#: individually, which is correct for both.
_PRE_NORMALIZED_DIMENSION = 3072


class GeminiEmbedder:
    """Hosted embeddings from Google's Generative Language API."""

    def __init__(
        self,
        *,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
        api_key: str | None = None,
        max_batch_size: int = 100,
        timeout: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "no API key: pass api_key= or set GEMINI_API_KEY. "
                "Use HashingEmbedder for tests — it needs neither."
            )
        self._api_key = key
        self._model = model
        self.dimension = dimension
        self.id = f"{model}@{dimension}"
        self.max_batch_size = max_batch_size
        # 001 only pre-normalizes at 3072; the index normalizes either way.
        self.produces_normalized = dimension == _PRE_NORMALIZED_DIMENSION
        self._timeout = timeout
        self._max_retries = max_retries

    async def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = await self._request(texts, task_type="RETRIEVAL_DOCUMENT")
        return normalize_rows(vectors)

    async def embed_query(self, text: str) -> np.ndarray:
        # Separate from embed_documents: the task type genuinely differs.
        vectors = await self._request([text], task_type="RETRIEVAL_QUERY")
        return normalize_row(vectors[0])

    async def _request(self, texts: list[str], *, task_type: str) -> np.ndarray:
        if len(texts) > self.max_batch_size:
            raise ValueError(
                f"batch of {len(texts)} exceeds max_batch_size {self.max_batch_size}; "
                f"the index batches for you, so this is a caller bug"
            )

        payload: dict[str, Any] = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    # Wrapped individually — see the note above.
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self.dimension,
                }
                for text in texts
            ]
        }
        url = _ENDPOINT.format(model=self._model)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await self._send_with_retry(client, url, payload)

        body = response.json()
        embeddings = body.get("embeddings")
        if embeddings is None or len(embeddings) != len(texts):
            got = "null" if embeddings is None else str(len(embeddings))
            raise RuntimeError(
                f"{self._model} returned {got} embeddings for {len(texts)} inputs. "
                f"If this says 1, the API aggregated the batch into a single vector."
            )
        matrix = np.asarray([item["values"] for item in embeddings], dtype=np.float32)
        if matrix.shape[1] != self.dimension:
            raise RuntimeError(
                f"{self._model} returned {matrix.shape[1]} dimensions, expected {self.dimension}"
            )
        return matrix

    async def _send_with_retry(
        self, client: httpx.AsyncClient, url: str, payload: dict[str, Any]
    ) -> httpx.Response:
        """Retry on 429 and 5xx with exponential backoff.

        Retry belongs to the embedder; the index has no idea what a rate limit is.
        """
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = await client.post(
                    url, json=payload, headers={"x-goog-api-key": self._api_key}
                )
                if response.status_code < 400:
                    return response
                if response.status_code == 429 or response.status_code >= 500:
                    last = RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
                else:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")
            except httpx.TransportError as exc:
                last = exc
            if attempt < self._max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"embedding request failed after {self._max_retries} attempts: {last}")
