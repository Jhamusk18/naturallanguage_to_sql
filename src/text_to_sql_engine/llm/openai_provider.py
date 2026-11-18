"""OpenAI chat-completions adapter behind the LLMProvider protocol."""

from __future__ import annotations

from typing import Any, cast

from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAIProvider:
    """Thin adapter over the OpenAI chat completions API.

    Temperature defaults to 0 — SQL generation should be as deterministic as
    the provider allows.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5)
        self._model = model
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        # Token accounting, read by the evaluation harness to report run cost.
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """One tool-calling turn: the model must call one of ``tools``."""
        effort = self._reasoning_effort
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, messages),
                tools=cast(Any, tools),
                tool_choice="required",
                temperature=self._temperature,
                extra_body={"reasoning_effort": effort} if effort else None,
            )
        except BadRequestError as error:
            # Groq rejects a reply that called no tool and returns the text it
            # generated; hand that text back so the loop can still use it.
            body = error.body if isinstance(error.body, dict) else {}
            inner = body.get("error", body) if isinstance(body, dict) else {}
            failed = inner.get("failed_generation") if isinstance(inner, dict) else None
            if failed is None:
                raise
            self.calls += 1
            return {"content": str(failed), "tool_calls": []}
        self._count(response)
        message = response.choices[0].message
        calls = []
        for call in message.tool_calls or []:
            function = getattr(call, "function", None)
            if function is not None:
                calls.append(
                    {"id": call.id, "name": function.name, "arguments": function.arguments}
                )
        return {"content": message.content, "tool_calls": calls}

    def _count(self, response: Any) -> None:
        self.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        messages: list[ChatCompletionMessageParam] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        effort = self._reasoning_effort
        extra_body = {"reasoning_effort": effort} if effort else None
        response = self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, messages),
            temperature=self._temperature,
            extra_body=extra_body,
        )
        self._count(response)
        return response.choices[0].message.content or ""
