"""A recording OpenAI client, for evaluation only.

The chain contains four uncached model calls -- intent classification,
extraction, follow-up wording, confirmation -- and any of them can flip. Measured
chain-against-chain, that made roughly 15% of conversations non-reproducible,
which is enough to make a strict migration criterion meaningless: a perfect
reimplementation would fail it, and a real bug would look the same.

This wraps the client rather than changing any of those call sites, because
every model call in the application already goes through an injected client.
Production constructs a real one in app.py and is untouched by this file; the
evaluation harness constructs this instead. The application cannot tell the
difference, which is the point -- caching inside the application would be a
behaviour change, and Part B forbids those.

Cached to disk, not just memory. In-process caching would make the equivalence
harness reproducible, since it runs both orchestrators in one process, but the
benchmarks run separately and the prize is reproducibility across runs and
machines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[3] / "reports" / "equivalence" / "model_cache.json"
)


def _key(model: str, messages: list, temperature: float, extra: dict) -> str:
    """Hash everything that can change the response.

    Temperature and max_tokens are included because two calls differing only in
    those are genuinely different calls; leaving them out would serve a cached
    answer to a question nobody asked.
    """

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "extra": {k: v for k, v in sorted(extra.items()) if k != "stream"},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return round(100.0 * self.hits / self.total, 1) if self.total else 0.0

    def describe(self) -> str:
        return f"{self.hits} hits, {self.misses} misses ({self.hit_rate}% cached)"


class _CachedMessage:
    def __init__(self, content: str | None):
        self.content = content


class _CachedChoice:
    def __init__(self, content: str | None):
        self.message = _CachedMessage(content)


class _CachedUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _CachedResponse:
    """Enough of the OpenAI response shape for every call site in this repo."""

    def __init__(self, record: dict):
        self.choices = [_CachedChoice(record.get("content"))]
        self.usage = _CachedUsage(
            record.get("prompt_tokens", 0), record.get("completion_tokens", 0)
        )


@dataclass
class CachingChatClient:
    """Wraps an OpenAI client, replaying identical calls from disk.

    A miss falls through to the wrapped client and is recorded. When `offline`
    is set a miss raises instead, which is what makes a run provably
    reproducible: it cannot quietly reach the network and return something new.
    """

    inner: Any = None
    path: Path = DEFAULT_CACHE
    offline: bool = False
    stats: CacheStats = field(default_factory=CacheStats)
    _cache: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        self.chat = _ChatNamespace(self)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._cache, indent=1, sort_keys=True), encoding="utf-8"
        )

    def _create(self, model: str, messages: list, **kwargs):
        temperature = kwargs.get("temperature", 1.0)
        key = _key(model, messages, temperature, kwargs)

        record = self._cache.get(key)
        if record is not None:
            self.stats.hits += 1
            return _CachedResponse(record)

        self.stats.misses += 1
        if self.offline:
            # Loud on purpose. A silent miss is a live call inside a run that
            # claims to be reproducible, which is worse than no cache at all.
            raise RuntimeError(
                f"cache miss in offline mode: {model} / {len(messages)} messages. "
                "Re-record with offline=False before comparing runs."
            )
        if self.inner is None:
            raise RuntimeError("no inner client to fall through to")

        response = self.inner.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        usage = getattr(response, "usage", None)
        self._cache[key] = {
            "content": response.choices[0].message.content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return response


class _Completions:
    def __init__(self, owner: CachingChatClient):
        self._owner = owner

    def create(self, model: str, messages: list, **kwargs):
        return self._owner._create(model, messages, **kwargs)


class _ChatNamespace:
    def __init__(self, owner: CachingChatClient):
        self.completions = _Completions(owner)
