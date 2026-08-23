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

**Both `create` and `parse` are intercepted, and the constructor refuses to
build if the wrapped client exposes a completions method this does not cover.**
An earlier version wrapped only `create`. Extraction uses `parse` with a
`response_format`, so every extraction call raised, was swallowed by the
extractor's own bounded-retry fallback, and returned "I couldn't reliably
capture that information" — plausible output from a broken client. The
equivalence runs then passed because both orchestrators were crippled
identically, and the chain looked newly deterministic because the component
measured as its source of non-determinism had been switched off. Partial
interception is worse than none: it produces confident, wrong measurements.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CACHE = (
    Path(__file__).resolve().parents[3] / "reports" / "partb" / "model_cache.json"
)


def _hashable(value: Any) -> Any:
    """Reduce a kwarg to something whose identity is what actually matters.

    `response_format` is a Pydantic class. Falling back to `str` would hash it as
    its name, so adding a field to the extraction schema would leave the key
    unchanged and the replay would rebuild the *new* model from JSON recorded
    against the old one -- a stale cache that looks like a hit. Hashing the JSON
    schema instead means a schema change invalidates exactly the affected
    entries.
    """

    schema = getattr(value, "model_json_schema", None)
    if callable(schema):
        return {"__schema__": schema()}
    return value


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
            "extra": {
                k: _hashable(v)
                for k, v in sorted(extra.items())
                if k != "stream"
            },
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


class _ParsedMessage:
    def __init__(self, content: str | None, parsed: Any):
        self.content = content
        self.parsed = parsed


class _ParsedChoice:
    def __init__(self, content: str | None, parsed: Any):
        self.message = _ParsedMessage(content, parsed)


class _CachedResponse:
    """Enough of the OpenAI response shape for every call site in this repo."""

    def __init__(self, record: dict, response_format: Any = None):
        content = record.get("content")
        if response_format is not None:
            # `parse` callers read `.message.parsed` and ignore `.content`.
            # The parsed object is rebuilt from the stored JSON through the same
            # model the live call would have validated against, so a cached
            # result cannot be shaped differently from a live one.
            parsed = (
                response_format.model_validate_json(content)
                if content is not None
                else None
            )
            self.choices = [_ParsedChoice(content, parsed)]
        else:
            self.choices = [_CachedChoice(content)]
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

    # Every completions method this wrapper implements. The guard below compares
    # against what the real client exposes, so a future SDK method cannot be
    # silently un-intercepted the way `parse` was.
    # "Covered" means intercepted, not necessarily supported. `stream` is
    # covered by refusing it: nothing in this application streams, and a
    # streaming call appearing later must fail loudly rather than slip past the
    # cache inside a run that reports itself fully cached.
    COVERED = ("create", "parse", "stream")

    def __post_init__(self):
        if self.path.exists():
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        self._assert_full_coverage()
        self.chat = _ChatNamespace(self)

    def _assert_full_coverage(self) -> None:
        """Refuse to run if the real client has a call path this cannot record."""

        if self.inner is None:
            return
        try:
            completions = self.inner.chat.completions
        except AttributeError:  # pragma: no cover - not an OpenAI-shaped client
            return
        uncovered = {
            name
            for name in ("create", "parse", "stream")
            if hasattr(completions, name) and name not in self.COVERED
        }
        if uncovered:
            raise RuntimeError(
                f"CachingChatClient does not intercept {sorted(uncovered)}. "
                "Add them to COVERED and implement them, or calls will reach the "
                "network inside a run that reports itself as fully cached."
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._cache, indent=1, sort_keys=True), encoding="utf-8"
        )

    def _call(self, method: str, model: str, messages: list, **kwargs):
        temperature = kwargs.get("temperature", 1.0)
        response_format = kwargs.get("response_format")
        # The method is part of the key: `create` and `parse` on the same
        # messages are different calls returning differently shaped results.
        key = _key(model, messages, temperature, {**kwargs, "_method": method})

        record = self._cache.get(key)
        if record is not None:
            self.stats.hits += 1
            return _CachedResponse(record, response_format)

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

        response = getattr(self.inner.chat.completions, method)(
            model=model, messages=messages, **kwargs
        )
        message = response.choices[0].message
        content = message.content
        if response_format is not None and content is None:
            # Some SDK paths populate only `.parsed`. Store its JSON so the
            # replay can rebuild the same object.
            parsed = getattr(message, "parsed", None)
            content = parsed.model_dump_json() if parsed is not None else None
        usage = getattr(response, "usage", None)
        self._cache[key] = {
            "content": content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        return response


class _Completions:
    def __init__(self, owner: CachingChatClient):
        self._owner = owner

    def create(self, model: str, messages: list, **kwargs):
        return self._owner._call("create", model, messages, **kwargs)

    def parse(self, model: str, messages: list, **kwargs):
        return self._owner._call("parse", model, messages, **kwargs)

    def stream(self, *args, **kwargs):
        raise RuntimeError(
            "streaming is not recorded. Nothing in this application streams; if "
            "that changes, implement it here rather than letting the call reach "
            "the network during an equivalence run."
        )


class _ChatNamespace:
    def __init__(self, owner: CachingChatClient):
        self.completions = _Completions(owner)
