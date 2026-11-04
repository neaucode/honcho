"""sync_turn tags each stored message with the Hermes session and state.db row it came from, and
honcho_search labels a hit with them, so a hit can be traced back to the exact Hermes message."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_plugin_honcho import HonchoMemoryProvider
from hermes_plugin_honcho.client import HonchoClientConfig
from hermes_plugin_honcho.session import HonchoSession, HonchoSessionManager

HERMES_SESSION = "20261103_101500_a1b2c3"
USER_TEXT = "Now decode it back."
REPLY_TEXT = "Decoding round-trips for every shape."


def _transcript(final_row_id=16):
    """The persisted transcript the harness hands sync_turn (each row stamped with its state.db
    ``_row_id``): an earlier turn, then this turn's opening message, a tool round, a mid-turn steer
    and the final reply."""
    final = {"role": "assistant", "content": REPLY_TEXT}
    if final_row_id is not None:
        final["_row_id"] = final_row_id
    return [
        {"role": "user", "content": "Add the ShapeIndex encoder.", "_row_id": 10},
        {"role": "assistant", "content": "Added.", "_row_id": 11},
        {"role": "user", "content": USER_TEXT, "_row_id": 12},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "_row_id": 13},
        {"role": "tool", "content": "ok", "tool_call_id": "c1", "_row_id": 14},
        {"role": "user", "content": "Also cover empty input.", "_row_id": 15},
        final,
    ]


def _manager() -> HonchoSessionManager:
    cfg = SimpleNamespace(
        write_frequency="turn", dialectic_reasoning_level="low", dialectic_dynamic=True, dialectic_max_chars=600,
        observation_mode="directional", user_observe_me=True, user_observe_others=True, ai_observe_me=True,
        ai_observe_others=True, message_max_chars=25000, dialectic_max_input_chars=10000,
    )
    return HonchoSessionManager(honcho=SimpleNamespace(), config=cfg)


def _session() -> HonchoSession:
    return HonchoSession(key="k", user_peer_id="user", assistant_peer_id="assistant", honcho_session_id="hs-k")


def _stored_turn(messages) -> HonchoSession:
    """Run one sync_turn and return the local session it wrote into."""
    session = _session()
    manager = MagicMock()
    manager.get_or_create.return_value = session
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(save_messages=True)
    provider._manager = manager
    provider._session_key = "k"
    provider._session_initialized = True
    provider.sync_turn(USER_TEXT, REPLY_TEXT, session_id=HERMES_SESSION, messages=messages)
    if provider._sync_thread is not None:
        provider._sync_thread.join(timeout=5)
    return session


def _flushed(session: HonchoSession) -> list:
    """Flush ``session`` through the manager and return every ``(content, metadata)`` peer.message got."""
    sent = []
    peer = SimpleNamespace(message=lambda content, metadata=None: sent.append((content, metadata)) or content)
    mgr = _manager()
    mgr._get_or_create_peer = lambda peer_id: peer
    mgr._sessions_cache[session.honcho_session_id] = SimpleNamespace(add_messages=lambda messages: None)
    assert mgr._flush_session(session) is True
    return sent


def test_turn_ids_reach_peer_message_as_metadata():
    sent = _flushed(_stored_turn(_transcript()))

    assert sent == [
        (USER_TEXT, {"hermes_session_id": HERMES_SESSION, "hermes_message_id": 12}),
        (REPLY_TEXT, {"hermes_session_id": HERMES_SESSION, "hermes_message_id": 16}),
    ]


def test_no_message_ids_when_the_final_reply_has_no_row():
    """The harness failed to save the reply: no row ids at all, rather than ids from an earlier turn."""
    sent = _flushed(_stored_turn(_transcript(final_row_id=None)))

    assert [content for content, _ in sent] == [USER_TEXT, REPLY_TEXT]
    for _content, metadata in sent:
        assert "hermes_message_id" not in (metadata or {})


def test_search_hits_carry_the_hermes_ids():
    hit = SimpleNamespace(content=USER_TEXT, peer_id="user", session_id="hs-k",
                          metadata={"hermes_session_id": HERMES_SESSION, "hermes_message_id": 12})
    legacy = SimpleNamespace(content="An older turn.", peer_id="user", session_id="hs-old", metadata={})
    honcho_client = SimpleNamespace(search=lambda q, filters=None, limit=10: [hit, legacy])
    mgr = _manager()
    mgr._cache["k"] = _session()

    with patch.object(HonchoSessionManager, "honcho", new_callable=lambda: property(lambda s: honcho_client)):
        result = mgr.search_context("k", "decode")

    assert f"[user · hs-k · hermes_session_id={HERMES_SESSION} hermes_message_id=12] {USER_TEXT}" in result
    assert "[user · hs-old] An older turn." in result
