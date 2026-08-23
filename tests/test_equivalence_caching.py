"""Unit tests for the recording client used by the Part B equivalence harness.

These exist because of a specific failure. The wrapper originally intercepted
only `chat.completions.create`. Extraction calls `chat.completions.parse`, so
every extraction raised inside a run that reported itself fully cached, the
extractor's own fallback turned that into "I couldn't reliably capture that
information", and the harness happily declared the two orchestrators equivalent
-- they were, in the sense that both were equally broken.

Nothing in the suite could have caught it, because the tests covered what the
wrapper did rather than what the application asks of it. The last test here is
the one that would have: it reads the real call sites and fails if any of them
names a method the wrapper does not implement.

Offline and unpaid: the inner client is a fake throughout.
"""

import re
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from pydantic import BaseModel  # noqa: E402

from evaluators.equivalence.caching_client import (  # noqa: E402
    CachingChatClient,
    _key,
)


class Extraction(BaseModel):
    reason: str
    urgency: int


class FakeMessage:
    def __init__(self, content, parsed=None):
        self.content = content
        self.parsed = parsed


class FakeUsage:
    prompt_tokens = 11
    completion_tokens = 3


class FakeResponse:
    def __init__(self, content, parsed=None):
        self.choices = [type("Choice", (), {"message": FakeMessage(content, parsed)})()]
        self.usage = FakeUsage()


class FakeCompletions:
    """Shaped like the OpenAI SDK, including the `stream` the wrapper refuses."""

    def __init__(self):
        self.create_calls = 0
        self.parse_calls = 0

    def create(self, model, messages, **kwargs):
        self.create_calls += 1
        return FakeResponse(f"reply {self.create_calls}")

    def parse(self, model, messages, **kwargs):
        self.parse_calls += 1
        parsed = Extraction(reason="sore throat", urgency=self.parse_calls)
        return FakeResponse(parsed.model_dump_json(), parsed)

    def stream(self, *args, **kwargs):  # pragma: no cover - never called
        raise AssertionError("the wrapper must refuse this before it is reached")


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions()


def client(tmp: Path, inner=None, offline=False):
    return CachingChatClient(inner=inner or FakeClient(), path=tmp, offline=offline)


class CoverageGuardTests(unittest.TestCase):
    """Silent partial interception is the failure; refuse to construct instead."""

    def test_a_client_with_an_uncovered_method_is_rejected(self):
        class Extended(FakeClient):
            def __init__(self):
                super().__init__()
                self.chat.completions.embed = lambda **k: None

        # Simulate a future SDK method by naming one the wrapper does not list.
        original = CachingChatClient.COVERED
        try:
            CachingChatClient.COVERED = ("create",)
            with self.assertRaises(RuntimeError) as caught:
                client(Path("/tmp/never-written.json"), inner=FakeClient())
        finally:
            CachingChatClient.COVERED = original

        self.assertIn("parse", str(caught.exception))

    def test_the_real_sdk_shape_is_accepted(self):
        self.assertIsNotNone(client(Path("/tmp/never-written.json")))

    def test_streaming_is_covered_by_refusing_it(self):
        wrapper = client(Path("/tmp/never-written.json"))

        with self.assertRaises(RuntimeError) as caught:
            wrapper.chat.completions.stream(model="m", messages=[])

        self.assertIn("not recorded", str(caught.exception))


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.path = Path("/tmp/equivalence-parse-cache.json")
        self.path.unlink(missing_ok=True)
        self.inner = FakeClient()

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_parse_reaches_the_inner_client_on_a_miss(self):
        wrapper = client(self.path, inner=self.inner)

        response = wrapper.chat.completions.parse(
            model="m", messages=[{"role": "user", "content": "hi"}],
            response_format=Extraction,
        )

        self.assertEqual(response.choices[0].message.parsed.reason, "sore throat")
        self.assertEqual(self.inner.chat.completions.parse_calls, 1)

    def test_a_replayed_parse_rebuilds_the_model_and_does_not_call_out(self):
        """The bug in one test: a cached parse must still yield `.parsed`."""

        wrapper = client(self.path, inner=self.inner)
        messages = [{"role": "user", "content": "hi"}]
        wrapper.chat.completions.parse(
            model="m", messages=messages, response_format=Extraction
        )
        wrapper.save()

        replay = client(self.path, inner=self.inner, offline=True)
        response = replay.chat.completions.parse(
            model="m", messages=messages, response_format=Extraction
        )

        parsed = response.choices[0].message.parsed
        self.assertIsInstance(parsed, Extraction)
        self.assertEqual(parsed.reason, "sore throat")
        self.assertEqual(parsed.urgency, 1)
        self.assertEqual(self.inner.chat.completions.parse_calls, 1)
        self.assertEqual(replay.stats.hits, 1)
        self.assertEqual(replay.stats.misses, 0)

    def test_create_and_parse_on_identical_messages_are_different_entries(self):
        """Same messages, different call: one must not serve the other's answer."""

        wrapper = client(self.path, inner=self.inner)
        messages = [{"role": "user", "content": "hi"}]

        wrapper.chat.completions.create(model="m", messages=messages)
        response = wrapper.chat.completions.parse(
            model="m", messages=messages, response_format=Extraction
        )

        self.assertEqual(response.choices[0].message.parsed.reason, "sore throat")
        self.assertEqual(wrapper.stats.hits, 0)
        self.assertEqual(wrapper.stats.misses, 2)

    def test_a_changed_schema_invalidates_the_entry(self):
        """Otherwise the replay rebuilds a new model from JSON recorded against the old one."""

        class Widened(BaseModel):
            reason: str
            urgency: int
            severity: str

        messages = [{"role": "user", "content": "hi"}]
        before = _key("m", messages, 0.0, {"response_format": Extraction})
        after = _key("m", messages, 0.0, {"response_format": Widened})

        self.assertNotEqual(before, after)

    def test_offline_mode_raises_rather_than_calling_out(self):
        wrapper = client(self.path, inner=self.inner, offline=True)

        with self.assertRaises(Exception):
            wrapper.chat.completions.parse(
                model="m", messages=[{"role": "user", "content": "new"}],
                response_format=Extraction,
            )
        self.assertEqual(self.inner.chat.completions.parse_calls, 0)


class CallSiteTests(unittest.TestCase):
    """The test that would have caught the original bug."""

    def test_every_client_method_the_application_calls_is_intercepted(self):
        used = set()
        for path in SRC.rglob("*.py"):
            if "evaluators" in path.parts:
                continue
            used.update(
                re.findall(
                    r"\bclient\.chat\.completions\.(\w+)\(", path.read_text(encoding="utf-8")
                )
            )

        self.assertTrue(used, "no client call sites found -- the pattern has drifted")
        uncovered = used - set(CachingChatClient.COVERED)
        self.assertEqual(
            uncovered,
            set(),
            f"the application calls {sorted(uncovered)}, which the recording client "
            "does not intercept. Equivalence runs would silently make live calls "
            "or fail inside a run reporting itself as cached.",
        )


if __name__ == "__main__":
    unittest.main()
