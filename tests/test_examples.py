"""Run the examples against the real Router API without provider calls."""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from model_router import Limits, ModelInfo, NoModelFitsError, RouterError

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FAKE_KEY = "offline-example-key"
MODEL_IDS = ("anthropic/claude-opus-5.5", "openai/gpt-6-luna", "google/gemini-3.5-flash-lite")
CATALOG = {name: ModelInfo(name, 32_000, 4_000, 1e-8, 1e-8, "Offline test model") for name in MODEL_IDS}


def load_example(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline(monkeypatch, tmp_path):
    # Never load a developer's .env, keys or a cached example module.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.delenv("jev_api_key", raising=False)
    with (
        patch("socket.socket.connect", side_effect=AssertionError("unexpected network call")),
        patch("model_router.router.fetch_catalog", return_value=CATALOG),
        patch(
            "model_router.jev.request_json",
            return_value={"answers": {"model": {"choice": "m0"}}},
        ) as route,
    ):
        yield route


def test_basic_main_runs_offline(offline, monkeypatch, capsys):
    basic = load_example("basic", monkeypatch)

    basic.main()

    output = capsys.readouterr().out
    assert "candidates from providers:" in output
    assert "<- cost-capped" in output
    assert "no fit:" in output
    assert offline.call_count == 3
    assert all(call.args[1] == FAKE_KEY for call in offline.call_args_list)


def test_agent_main_runs_offline(offline, monkeypatch, capsys):
    load_example("basic", monkeypatch)  # agent.py imports its neighboring example
    agent = load_example("agent", monkeypatch)
    replies = ["Research\nDraft\nReview", "research", "draft", "review", "summary", "Canberra", "algorithm"]
    responses = [{"choices": [{"message": {"content": reply}}]} for reply in replies]
    with patch.object(agent, "request_json", side_effect=responses) as chat:
        agent.main()

    output = capsys.readouterr().out
    assert "estimated spend" in output
    assert "== Chat session" in output
    assert "Canberra" in output and "algorithm" in output
    assert chat.call_count == offline.call_count == 7
    assert all(call.args[0] == agent.CHAT_URL and call.args[1] == FAKE_KEY for call in chat.call_args_list)
    assert all(call.args[2]["model"] in CATALOG for call in chat.call_args_list)


def clear_env_keys(monkeypatch, *keys):
    """Remove keys from os.environ while guaranteeing the originals are restored afterwards."""
    for key in keys:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)


def test_load_env_parses_keys_and_ignores_comments_and_blanks(monkeypatch, tmp_path):
    basic = load_example("basic", monkeypatch)
    monkeypatch.chdir(tmp_path)
    clear_env_keys(monkeypatch, "OPENROUTER_API_KEY", "JEV_API_KEY")
    (tmp_path / ".env").write_text(
        "# a comment\n\nOPENROUTER_API_KEY='sk-or-file'\nJEV_API_KEY=\"sk-jev-file\"\nNOT_AN_ASSIGNMENT\n"
    )
    basic.load_env()
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-file"
    assert os.environ["JEV_API_KEY"] == "sk-jev-file"
    assert "NOT_AN_ASSIGNMENT" not in os.environ


def test_load_env_does_not_override_existing_environment(monkeypatch, tmp_path):
    basic = load_example("basic", monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-already-set")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-file\n")
    basic.load_env()
    assert os.environ["OPENROUTER_API_KEY"] == "sk-already-set"


def test_routing_key_falls_back_to_jev_key(monkeypatch):
    basic = load_example("basic", monkeypatch)
    clear_env_keys(monkeypatch, "OPENROUTER_API_KEY", "JEV_API_KEY", "jev_api_key")
    monkeypatch.setenv("JEV_API_KEY", "sk-jev")
    assert basic.routing_key() == {"jev_api_key": "sk-jev"}


def test_routing_key_accepts_lowercase_jev_env_var(monkeypatch):
    basic = load_example("basic", monkeypatch)
    clear_env_keys(monkeypatch, "OPENROUTER_API_KEY", "JEV_API_KEY", "jev_api_key")
    monkeypatch.setenv("jev_api_key", "sk-jev-lower")
    assert basic.routing_key() == {"jev_api_key": "sk-jev-lower"}


def test_routing_key_exits_when_no_key_is_set(monkeypatch):
    basic = load_example("basic", monkeypatch)
    clear_env_keys(monkeypatch, "OPENROUTER_API_KEY", "JEV_API_KEY", "jev_api_key")
    with pytest.raises(SystemExit) as ctx:
        basic.routing_key()
    assert "OPENROUTER_API_KEY" in str(ctx.value)


def test_agent_pick_falls_back_when_router_is_unreachable(monkeypatch, capsys):
    load_example("basic", monkeypatch)  # agent.py imports its neighboring example
    agent = load_example("agent", monkeypatch)
    router = MagicMock()
    router.route.side_effect = RouterError("Jev unreachable")
    assert agent.pick(router, "hello", Limits()) == agent.FALLBACK
    assert "router unavailable" in capsys.readouterr().out


def test_agent_pick_lets_no_model_fits_error_through(monkeypatch):
    load_example("basic", monkeypatch)
    agent = load_example("agent", monkeypatch)
    router = MagicMock()
    router.route.side_effect = NoModelFitsError("input too large")
    with pytest.raises(NoModelFitsError):
        agent.pick(router, "hello", Limits())


def test_agent_chat_reports_empty_reasoning_replies(monkeypatch):
    load_example("basic", monkeypatch)
    agent = load_example("agent", monkeypatch)
    reply = {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    with patch.object(agent, "request_json", return_value=reply):
        assert agent.chat("key", "model", [], 100) == "<empty reply, finish_reason=length>"


def test_agent_plan_and_execute_skips_cost_tracking_for_fallback_model(monkeypatch, capsys):
    load_example("basic", monkeypatch)
    agent = load_example("agent", monkeypatch)
    router = MagicMock()
    router.models = [ModelInfo("acme/local", 100_000, 4_000, 1e-8, 1e-8, "stub model")]
    router.route.side_effect = RouterError("Jev unreachable")
    replies = ["plan it\nwrite it\nship it", "r1", "r2", "r3", "the summary"]
    responses = [{"choices": [{"message": {"content": r}}]} for r in replies]
    with patch.object(agent, "request_json", side_effect=responses) as chat:
        agent.plan_and_execute(router, "offline-key", "launch the tool")
    assert all(call.args[2]["model"] == agent.FALLBACK for call in chat.call_args_list)
    assert "estimated spend $0.0000" in capsys.readouterr().out
