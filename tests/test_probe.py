"""Readiness probe on the run's real call paths (READINESS_AUDIT §3), snapshot pins and the live preflight check.

The probe runs against a local fake OpenAI server, so the real code paths are exercised unchanged: the OpenAI
SDK, Inspect's own OpenAI provider (Responses API for GPT-6, chat completions for gpt-4o-mini) and `BuildLlm`.
The server answers per endpoint, with reasoning tokens that depend on the requested effort, a configurable
served snapshot per alias, and x-ratelimit headers. No network, no key, no spend.
"""

import asyncio
import datetime as dt
import json
import shutil
import threading
import warnings
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ape import snapshots
from ape.config import ROOT
from ape.llm.ledger import Ledger
from ape.models import PreflightError, SnapshotWarning, load_profile, preflight, require_preflight

from test_smoke import _load, sc  # readiness/ is on sys.path via test_smoke

probe_mod = _load("probe_openai")

EFFORT_TOKENS = {"high": 900, "medium": 400, "low": 120, "minimal": 10, "none": 0}


class FakeOpenAI:
    """Per-endpoint fake: /v1/models, /v1/responses, /v1/chat/completions, /v1/embeddings."""

    def __init__(self) -> None:
        self.snapshots = {"gpt-6-luna": "gpt-6-luna-2026-08-14", "gpt-6-sol": "gpt-6-sol-2026-07-30", "gpt-4o-mini": "gpt-4o-mini-2024-07-18", "text-embedding-3-small": "text-embedding-3-small"}
        self.effort_tokens = dict(EFFORT_TOKENS)  # tests flatten this to make effort "ignored"
        self.reject_effort: set[str] = set()  # aliases that 400 on any reasoning effort
        self.no_reasoning_field = False
        self.requests: list[tuple[str, dict]] = []

    def respond(self, path: str, body: dict) -> tuple[int, dict]:
        self.requests.append((path, body))
        model = body.get("model", "")
        served = self.snapshots.get(model, model)
        if path.endswith("/responses"):
            effort = (body.get("reasoning") or {}).get("effort")
            schema = ((body.get("text") or {}).get("format") or {}).get("type") == "json_schema"
        elif path.endswith("/chat/completions"):
            effort = body.get("reasoning_effort")
            schema = (body.get("response_format") or {}).get("type") == "json_schema"
        elif path.endswith("/embeddings"):
            return 200, {"object": "list", "model": served, "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 8}], "usage": {"prompt_tokens": 2, "total_tokens": 2}}
        else:
            return 404, {"error": {"message": "not found"}}
        if effort and model in self.reject_effort:
            return 400, {"error": {"message": "Unsupported parameter: 'reasoning_effort' is not supported with this model.", "type": "invalid_request_error", "param": "reasoning_effort", "code": "unsupported_parameter"}}
        rt = self.effort_tokens.get(effort or "medium", 50)
        text = '{"ok": true}' if schema else "466"
        if path.endswith("/responses"):
            details = {} if self.no_reasoning_field else {"reasoning_tokens": rt}
            return 200, {
                "id": "resp_1", "object": "response", "created_at": 1_790_000_000, "status": "completed", "model": served,
                "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": 30, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": rt + 5, "output_tokens_details": details, "total_tokens": rt + 35},
                "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "error": None, "incomplete_details": None, "instructions": None, "metadata": {},
                "temperature": None, "top_p": None, "reasoning": {"effort": effort, "summary": None}, "text": {"format": {"type": "text"}},
            }
        details = {} if self.no_reasoning_field else {"reasoning_tokens": rt}
        return 200, {
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1_790_000_000, "model": served,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": rt + 5, "total_tokens": rt + 35, "completion_tokens_details": details, "prompt_tokens_details": {"cached_tokens": 0}},
        }


@pytest.fixture
def fake_openai(monkeypatch):
    fake = FakeOpenAI()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.send_header("x-ratelimit-limit-tokens", "2000000")
            self.send_header("x-ratelimit-limit-requests", "10000")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            ids = sorted(fake.snapshots)
            self._send(200, {"object": "list", "data": [{"id": i, "object": "model", "created": 0, "owned_by": "openai"} for i in ids]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["content-length"])) or b"{}")
            self._send(*fake.respond(self.path, body))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-tests")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}/v1")
    yield fake
    server.shutdown()


def _probe(profile, tmp_path: Path, **kwargs) -> dict:
    from openai import AsyncOpenAI, OpenAI

    return asyncio.run(probe_mod.run_probe(profile, client=OpenAI(), build_client=AsyncOpenAI(), ledger=Ledger(tmp_path / "ledger.jsonl"), **kwargs))


# ---------- the probe on the real paths ----------


def test_gate_profile_is_probed_on_the_responses_api_and_the_build_client(fake_openai, tmp_path):
    report = _probe(load_profile("gate"), tmp_path)
    assert report["version"] == sc.PROBE_VERSION and report["price_table"]["checked"] == "2026-09-30"
    roles = report["roles"]
    assert set(roles) == {"agent", "kg", "judge", "build"}
    # Inspect-path roles really went through the Responses API, the build role through chat completions.
    assert {r: list(e["paths"]) for r, e in roles.items()} == {"agent": ["inspect"], "kg": ["inspect"], "judge": ["inspect"], "build": ["build"]}
    assert all(roles[r]["paths"]["inspect"]["api"] == "responses" for r in ("agent", "kg", "judge"))
    assert roles["build"]["paths"]["build"]["api"] == "chat.completions"
    paths = [p for p, _ in fake_openai.requests]
    assert any(p.endswith("/responses") for p in paths) and any(p.endswith("/chat/completions") for p in paths)
    # Effort reached the server on both paths (reasoning.effort; reasoning_effort) and is honoured.
    agent = roles["agent"]["paths"]["inspect"]
    assert agent["effort"]["verdict"]["status"] == sc.PASS and agent["effort"]["verdict"]["measured"] == {"reasoning_tokens_high": 900, "reasoning_tokens_low": 120}
    assert roles["build"]["paths"]["build"]["effort"]["verdict"]["status"] == sc.PASS
    assert agent["as_configured"]["accepted"] and agent["structured_output"]["accepted"]
    # Effort calls run once per (path, model): agent, kg and judge share gpt-6-luna on the Inspect path.
    effort_calls = [b for p, b in fake_openai.requests if p.endswith("/responses") and (b.get("reasoning") or {}).get("effort") in ("high",) and "divisible" in json.dumps(b)]
    assert len(effort_calls) == 1
    # Snapshots per alias, across roles and paths; rate limits and embeddings.
    assert report["snapshots"]["gpt-6-luna"]["snapshot"] == "gpt-6-luna-2026-08-14"
    assert report["snapshots"]["gpt-6-luna"]["roles"] == ["agent (inspect)", "kg (inspect)", "judge (inspect)", "build (build)"]
    assert report["ratelimits"]["gpt-6-luna"]["headers"]["x-ratelimit-limit-tokens"] == "2000000"
    assert sc.ratelimit_headers(report, "openai/gpt-6-luna")["x-ratelimit-limit-tokens"] == "2000000"
    assert report["embeddings"]["dim"] == 8 and "gpt-6-luna" in report["models_available"]
    # Build-path probe calls are metered in the ledger, tagged as the probe's.
    entries = Ledger(tmp_path / "ledger.jsonl").read()
    assert entries and all(e.context["source"] == "probe" for e in entries)
    assert sc.effort_from_probe(report)["status"] == sc.PASS


def test_the_anchor_profile_uses_chat_completions_and_skips_effort_for_gpt_4o_mini(fake_openai, tmp_path):
    report = _probe(load_profile("anchor"), tmp_path)
    agent = report["roles"]["agent"]["paths"]["inspect"]
    assert agent["api"] == "chat.completions" and agent["effort"]["verdict"]["status"] == sc.SKIP
    assert report["roles"]["build"]["paths"]["build"]["effort"]["verdict"]["status"] == sc.SKIP
    # The judge's sampling settings reached the server as configured (temperature 0, seed 42).
    judge_calls = [b for p, b in fake_openai.requests if p.endswith("/chat/completions") and b.get("seed") == 42]
    assert judge_calls and all(b.get("temperature") == 0 for b in judge_calls)
    assert sc.effort_from_probe(report)["status"] == sc.PASS


def test_effort_that_changes_nothing_or_is_rejected_or_unreported_fails_with_a_reason(fake_openai, tmp_path):
    fake_openai.effort_tokens = {k: 64 for k in EFFORT_TOKENS}
    flat = sc.effort_from_probe(_probe(load_profile("gate"), tmp_path))
    assert flat["status"] == sc.FAIL and "agent/inspect: effort not honoured" in flat["reason"] and "high used 64 reasoning tokens, low 64" in flat["reason"]

    fake_openai.effort_tokens = dict(EFFORT_TOKENS)
    fake_openai.reject_effort = {"gpt-6-luna"}
    rejected = sc.effort_from_probe(_probe(load_profile("gate"), tmp_path))
    assert rejected["status"] == sc.FAIL and "rejected" in rejected["reason"] and "as configured call rejected" in rejected["reason"]

    fake_openai.reject_effort = set()
    fake_openai.no_reasoning_field = True
    silent = sc.effort_from_probe(_probe(load_profile("gate"), tmp_path))
    assert silent["status"] == sc.FAIL and "reports no reasoning tokens" in silent["reason"]


def test_effort_from_probe_needs_the_new_format_and_every_required_role(fake_openai, tmp_path):
    assert "predates the call-path probe" in sc.effort_from_probe({"agent": {"model": "m", "plain": {}}})["reason"]
    report = _probe(load_profile("gate"), tmp_path)
    del report["roles"]["kg"]
    assert "did not cover ['kg']" in sc.effort_from_probe(report)["reason"]


def test_cli_overrides_swap_a_roles_model_and_keep_its_effort():
    p = probe_mod.apply_overrides(load_profile("gate"), {"agent": "openai/gpt-6-sol", "build": None})
    assert p.roles["agent"].model == "openai/gpt-6-sol" and p.roles["agent"].reasoning_effort == "high"
    assert p.roles["build"] == load_profile("gate").roles["build"]


def test_cli_probe_writes_the_report_and_never_prints_the_key(fake_openai, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))  # the build ledger (Config().ledger_path) goes here
    out = tmp_path / "probe.json"
    probe_mod.main(["--profile", "gate", "--out", str(out)])
    report = json.loads(out.read_text())
    assert Ledger(tmp_path / "cache" / "ledger.jsonl").read(), "build-path probe calls are metered in the build ledger"
    assert report["version"] == sc.PROBE_VERSION and report["roles"]["agent"]["paths"]["inspect"]["api"] == "responses"
    printed = capsys.readouterr()
    assert "sk-fake-for-tests" not in printed.out + printed.err and '"check": "pass"' in printed.out


# ---------- pins ----------


def _report(snaps: dict[str, list[str]], probed_at: str = "2026-10-02T09:00:00+00:00") -> dict:
    return {
        "version": sc.PROBE_VERSION,
        "probed_at": probed_at,
        "profile": "gate",
        "price_table": {"checked": "2026-09-30"},
        "snapshots": {a: {"snapshots_seen": seen, "snapshot": seen[0] if len(set(seen)) == 1 else None, "roles": ["agent (inspect)"]} for a, seen in snaps.items()},
    }


@pytest.fixture
def provenance(tmp_path) -> Path:
    path = tmp_path / "PROVENANCE.md"
    shutil.copy(ROOT / "PROVENANCE.md", path)
    return path


def test_pin_writes_the_block_keeps_other_pins_and_records_superseded_ones(provenance, tmp_path):
    real = (ROOT / "PROVENANCE.md").read_text()
    assert snapshots.read_pins(provenance) == {}
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["gpt-6-luna-2026-08-14"], "gpt-6-sol": ["gpt-6-sol-2026-07-30"]})))
    res = probe_mod.pin(probe, provenance)
    assert res["pinned"] == {"gpt-6-luna": "gpt-6-luna-2026-08-14", "gpt-6-sol": "gpt-6-sol-2026-07-30"}
    assert snapshots.read_pins(provenance) == res["pinned"]
    # A later probe of one alias: that pin changes (and is kept as superseded), the other pin stays.
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["gpt-6-luna-2026-09-20"]})))
    res = probe_mod.pin(probe, provenance)
    assert res["changed"] == {"gpt-6-luna": ("gpt-6-luna-2026-08-14", "gpt-6-luna-2026-09-20")} and res["kept"] == ["gpt-6-sol"]
    assert snapshots.read_pins(provenance) == {"gpt-6-luna": "gpt-6-luna-2026-09-20", "gpt-6-sol": "gpt-6-sol-2026-07-30"}
    text = provenance.read_text()
    assert "`gpt-6-luna` was `gpt-6-luna-2026-08-14`, now `gpt-6-luna-2026-09-20`" in text and text.count(snapshots.BEGIN) == 1
    assert (ROOT / "PROVENANCE.md").read_text() == real, "tests never touch the real file"


def test_pin_refuses_a_missing_or_inconsistent_snapshot(provenance, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["a", "b"], "gpt-6-sol": []})))
    with pytest.raises(SystemExit, match=r"several snapshots(.|\n)*gpt-6-sol: no call reported"):
        probe_mod.pin(probe, provenance)
    assert snapshots.read_pins(provenance) == {}
    probe.write_text(json.dumps({"agent": {}}))
    with pytest.raises(SystemExit, match="predates the call-path probe"):
        probe_mod.pin(probe, provenance)


# ---------- live preflight ----------


def _env(tmp_path) -> Path:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test\n")
    return env


def test_live_preflight_refuses_a_changed_snapshot_passes_a_match_and_warns_without_pins(provenance, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("APE_BUILD_FALLBACK", raising=False)
    probe = tmp_path / "probe.json"
    today = dt.date.today().isoformat()
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["gpt-6-luna-2026-08-14"], "text-embedding-3-small": ["text-embedding-3-small"]}, probed_at=f"{today}T09:00:00+00:00")))
    kw = {"env_path": _env(tmp_path), "probe_path": probe, "provenance_path": provenance}

    # No pins yet: a warning, not a refusal.
    with pytest.warns(SnapshotWarning, match="no model snapshots are pinned"):
        assert preflight("gate", live=True, **kw) == []
    # Pinned and still served: go, silently.
    probe_mod.pin(probe, provenance)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SnapshotWarning)
        assert preflight("gate", live=True, **kw) == []
    # The alias moved to a new snapshot: refuse, naming both.
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["gpt-6-luna-2026-09-20"], "text-embedding-3-small": ["text-embedding-3-small"]}, probed_at=f"{today}T10:00:00+00:00")))
    with pytest.raises(PreflightError, match=r"gpt-6-luna: the latest probe was served by snapshot gpt-6-luna-2026-09-20, but PROVENANCE.md pins gpt-6-luna-2026-08-14"):
        require_preflight("gate", live=True, **kw)
    # Offline (dry) runs never look at snapshots.
    assert preflight("gate", live=False, **kw) == []


def test_snapshot_status_warns_on_unpinned_unchecked_and_old_probes(provenance, tmp_path):
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["s1"], "gpt-6-sol": ["s2"]})))
    probe_mod.pin(probe, provenance)
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["s1"]}, probed_at="2026-09-01T09:00:00+00:00")))
    status = snapshots.snapshot_status(probe, provenance, aliases={"gpt-6-luna", "gpt-6-sol", "gpt-6-astra"}, today=dt.date(2026, 10, 2))
    assert status["problems"] == []
    w = " ".join(status["warnings"])
    assert "gpt-6-astra: called by this run but not pinned" in w and "gpt-6-sol: pinned to s2 but the latest probe did not resolve it" in w and "31 days old" in w
    probe.unlink()
    assert "unreadable" in snapshots.snapshot_status(probe, provenance)["problems"][0]


def test_live_preflight_checks_only_the_fallback_builder_when_it_is_selected(provenance, tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(_report({"gpt-6-luna": ["s1"], "text-embedding-3-small": ["e"]}, probed_at=f"{dt.date.today().isoformat()}T09:00:00+00:00")))
    probe_mod.pin(probe, provenance)
    kw = {"env_path": _env(tmp_path), "probe_path": probe, "provenance_path": provenance}
    monkeypatch.delenv("APE_BUILD_FALLBACK", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SnapshotWarning)
        assert preflight("gate", live=True, **kw) == []
    monkeypatch.setenv("APE_BUILD_FALLBACK", "1")
    with pytest.warns(SnapshotWarning, match="gpt-6-sol: called by this run but not pinned"):
        preflight("gate", live=True, **kw)


def test_run_gate_live_preflight_refuses_a_moved_snapshot(provenance, tmp_path, monkeypatch):
    import os

    from ape.run_gate import GateRun, read_manifest, run_phases

    for k in [k for k in os.environ if k.startswith("APE_")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")  # nothing here makes a request
    probe = tmp_path / "openai_probe.json"
    report = _report({"gpt-6-luna": ["s1"], "text-embedding-3-small": ["e1"]}, probed_at=f"{dt.date.today().isoformat()}T09:00:00+00:00")
    report["models_available"] = ["gpt-6-luna", "gpt-6-sol", "text-embedding-3-small"]
    probe.write_text(json.dumps(report))
    probe_mod.pin(probe, provenance)
    run = GateRun("live-pins", runs_root=tmp_path / "runs", env_path=tmp_path / "no.env", probe_path=probe, provenance_path=provenance)
    assert run_phases(run, "preflight") == {"preflight": "done"}

    report["snapshots"]["gpt-6-luna"] = {"snapshot": "s2", "snapshots_seen": ["s2"], "roles": ["agent (inspect)"]}
    probe.write_text(json.dumps(report))
    with pytest.raises(PreflightError, match="gpt-6-luna: the latest probe was served by snapshot s2, but PROVENANCE.md pins s1"):
        run_phases(run, "preflight")
    assert read_manifest(run, "preflight")["status"] == "failed"


# ---------- smoke's effort and burst checks read the new format ----------


def test_smoke_dry_effort_check_runs_the_probe_code_offline():
    from types import SimpleNamespace

    c = {"args": SimpleNamespace(dry=True), "profile": load_profile("gate")}
    res, logs = probe_smoke().check_effort(c)
    assert res["status"] == sc.PASS and logs == []
    assert set(res["measured"]["paths"]) == {"agent/inspect", "kg/inspect", "judge/inspect", "build/build"}
    assert "dry" in res["measured"]["source"]


def test_smoke_live_effort_check_reads_the_probe_file(fake_openai, tmp_path, monkeypatch):
    from types import SimpleNamespace

    smoke = probe_smoke()
    report = _probe(load_profile("gate"), tmp_path)
    path = tmp_path / "openai_probe.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(smoke, "PROBE_PATH", path)
    res, _ = smoke.check_effort({"args": SimpleNamespace(dry=False), "profile": load_profile("gate")})
    assert res["status"] == sc.PASS and res["measured"]["snapshots"]["gpt-6-luna"] == "gpt-6-luna-2026-08-14"
    report["roles"]["kg"]["paths"]["inspect"]["effort"]["verdict"] = sc.result(sc.FAIL, {}, reason="flat")
    path.write_text(json.dumps(report))
    assert smoke.check_effort({"args": SimpleNamespace(dry=False), "profile": replace(load_profile("gate"))})[0]["status"] == sc.FAIL


def probe_smoke():
    from test_smoke import smoke

    return smoke
