"""Pre-LLM intent gate: requests to change data never reach the model.

Guardrails already make writes impossible, but a model asked to "drop a table"
tends to answer with a harmless SELECT, which looks like a successful query.
Rejecting the request up front is the honest outcome.
"""

from __future__ import annotations

import re

_VERBS = r"delete|drop|truncate|insert|update|alter|remove|erase|wipe|destroy|modify|grant|revoke"
_WRITE_INTENT = re.compile(
    rf"(?:^|[.;!?]\s*|\b(?:and|then|also|please|you)\s+)(?:{_VERBS})\b", re.IGNORECASE
)


def detect_write_intent(question: str) -> str | None:
    """The offending phrase when the question asks to modify data, else None."""
    match = _WRITE_INTENT.search(question.strip())
    return match.group(0).strip(" .;!?") if match else None
