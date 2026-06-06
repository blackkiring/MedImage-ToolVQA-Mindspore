from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    images: list[str] = field(default_factory=list)
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    stop: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...

    @abstractmethod
    def health_check(self) -> bool:
        ...


class MockLLMClient(LLMClient):
    """Returns canned responses for testing."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._default_vqa = json.dumps(
            {
                "question": "Based on this medical image, what is the most likely finding?",
                "options": [
                    "(A) Normal",
                    "(B) Abnormal mass",
                    "(C) Inconclusive",
                    "(D) Benign lesion",
                    "(E) Malignant tumor",
                ],
                "answer": "B",
            },
            ensure_ascii=False,
        )
        self._default_reasoning = (
            "<think>Examining the anatomical landmarks in the image to locate the region of interest.</think>"
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = request.prompt[:80]
        if key in self._responses:
            text = self._responses[key]
        elif "generate a medical" in request.prompt.lower() or "question" in request.prompt.lower():
            text = self._default_vqa
        else:
            text = self._default_reasoning

        return GenerationResult(
            text=text,
            model="mock",
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=10.0,
            request_id=str(uuid.uuid4()),
        )

    def health_check(self) -> bool:
        return True


class HttpLLMClient(LLMClient):
    """OpenAI-compatible HTTP API client."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "default",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import urllib.request
        import urllib.error

        start = time.monotonic()

        messages: list[dict[str, Any]] = []
        if request.images:
            content: list[dict[str, Any]] = []
            for img in request.images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": request.prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())

            latency = (time.monotonic() - start) * 1000
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return GenerationResult(
                text=choice["message"]["content"],
                model=data.get("model", request.model or self.model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=latency,
                request_id=data.get("id", ""),
            )
        except Exception as e:
            return GenerationResult(
                text="",
                model=request.model or self.model,
                error=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def health_check(self) -> bool:
        try:
            result = self.generate(GenerationRequest(prompt="ping", max_tokens=1))
            return result.error is None
        except Exception:
            return False
