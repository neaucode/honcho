"""dialecticCadence 0 turns automatic dialectic off: no spawn path and no sync recall reaches .chat()."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from hermes_plugin_honcho import recall_sync
from hermes_plugin_honcho.dialectic import DialecticMixin


class Provider(DialecticMixin):
    """The provider state the dialectic mixin and prefetch_sync read; counts dialectic runs."""

    def __init__(self, cadence: int):
        self._dialectic_cadence = cadence
        self._dialectic_empty_streak = 0
        self._last_dialectic_turn = -999
        self._last_context_turn = -999
        self._context_cadence = 1
        self._injection_frequency = "every-turn"
        self._dialectic_depth = 1
        self._query_rewriter = None
        self._query_rewrite_enabled = False
        self._prefetch_lock = threading.Lock()
        self._prefetch_result, self._prefetch_result_fired_at = "", -999
        self._prefetch_thread = None
        self._prefetch_thread_started_at = 0.0
        self._recall_sync_lock = threading.Lock()
        self._recall_sync_thread = None
        self._recall_generation = None
        self._session_key, self._turn_count = "s1", 1
        self._config = SimpleNamespace(timeout=5.0)
        self.runs: list[str] = []
        self.dialectic_queries: list[str] = []
        self._manager = SimpleNamespace(
            get_prefetch_context=lambda *a, **k: {"card": "base"},
            dialectic_query=lambda session, prompt, **k: self.dialectic_queries.append(prompt) or "answer",
        )

    def _run_dialectic_depth(self, query, use_query_rewrite=True):
        self.runs.append(query)
        return "answer"

    # prefetch_sync collaborators
    def _is_trivial_prompt(self, query):
        return False

    def _session_ready(self):
        return True

    def _resolve_pass_level(self, i, query=""):
        return "minimal"

    def _format_first_turn_context(self, ctx):
        return ctx.get("card", "")

    def _pop_auth_notice(self):
        return ""

    def _truncate_to_budget(self, text):
        return text


@pytest.mark.parametrize("label", ["dialectic prewarm", "first-turn dialectic", "prefetch"])
def test_cadence_zero_spawns_nothing(label):
    p = Provider(cadence=0)
    assert p._spawn_dialectic("q", thread_name="t", fired_at=1, log_label=label) is None
    assert p._prefetch_thread is None
    assert p.runs == []


def test_cadence_positive_still_spawns():
    p = Provider(cadence=1)
    thread = p._spawn_dialectic("q", thread_name="t", fired_at=1, log_label="prefetch")
    thread.join(timeout=2)
    assert p.runs == ["q"]
    assert p._last_dialectic_turn == 1


def test_sync_recall_cadence_zero_injects_base_and_skips_dialectic():
    p = Provider(cadence=0)
    for turn in range(1, 9):
        p._turn_count = turn
        assert recall_sync.prefetch_sync(p, "what have I been working on?") == "base"
    assert p.dialectic_queries == []


def test_sync_recall_cadence_positive_still_queries_dialectic():
    p = Provider(cadence=1)
    out = recall_sync.prefetch_sync(p, "what have I been working on?")
    assert len(p.dialectic_queries) == 1
    assert "answer" in out
