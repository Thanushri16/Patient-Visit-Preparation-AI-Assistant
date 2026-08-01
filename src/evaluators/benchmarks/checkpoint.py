"""Append-only checkpoint store so an interrupted benchmark run can resume.

Each completed scenario is written as one JSON line immediately after its batch
finishes. A resumed run replays the file, skips the test IDs it already holds,
and evaluates only what is left. Line-oriented storage means a process killed
mid-write loses at most the final record rather than the whole run.
"""

import json
from pathlib import Path
from typing import Any, Iterator


class CheckpointStore:
    """Read and append scenario results keyed by benchmark test ID."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        """Return completed results by test ID, ignoring any truncated tail."""

        if not self.path.exists():
            return {}
        results: dict[str, dict[str, Any]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line means the process died mid-write; the
                # scenario simply gets re-run.
                continue
            test_id = record.get("scenario", {}).get("test_id")
            if test_id:
                results[str(test_id)] = record
        return results

    def append_batch(self, results: list[dict[str, Any]]) -> None:
        """Durably append one batch of completed results."""

        if not results:
            return
        lines = [
            json.dumps(result, ensure_ascii=False, default=str) + "\n" for result in results
        ]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


def batched(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive slices of `items` of at most `size` elements."""

    if size < 1:
        raise ValueError("batch size must be at least 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]
