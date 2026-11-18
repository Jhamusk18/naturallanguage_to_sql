"""Retrieves the most relevant (question, SQL) exemplars for a new question.

Exemplar questions are embedded once at construction; each lookup embeds only
the incoming question and ranks by cosine similarity. Ties resolve by store
order, which keeps retrieval fully deterministic with the FakeEmbedder.
"""

from __future__ import annotations

from collections.abc import Sequence

from text_to_sql_engine.embeddings.base import Embedder, cosine_similarity
from text_to_sql_engine.models import Exemplar


class FewShotStore:
    def __init__(self, exemplars: Sequence[Exemplar], embedder: Embedder) -> None:
        self._exemplars = list(exemplars)
        self._embedder = embedder
        self._vectors: list[list[float]] = []
        self._space = ""
        self._refresh_vectors()

    def _refresh_vectors(self) -> None:
        if self._exemplars:
            self._vectors = self._embedder.embed([e.question for e in self._exemplars])
        self._space = str(getattr(self._embedder, "space", ""))

    def __len__(self) -> int:
        return len(self._exemplars)

    def top_k(self, question: str, k: int = 3) -> list[Exemplar]:
        """The k exemplars whose questions are most similar to this one."""
        if not self._exemplars or k <= 0:
            return []
        question_vector = self._embedder.embed([question])[0]
        if str(getattr(self._embedder, "space", "")) != self._space:
            self._refresh_vectors()
            question_vector = self._embedder.embed([question])[0]
        scored = sorted(
            enumerate(self._exemplars),
            key=lambda pair: (
                -cosine_similarity(question_vector, self._vectors[pair[0]]),
                pair[0],
            ),
        )
        return [exemplar for _index, exemplar in scored[:k]]
