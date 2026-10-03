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
    """Per-endpoint fake: /v1/models, /v1/responses, /v1/responses/compact, /v1/chat/completions, /v1/embeddings."""

    def __init__(self) -> None:
        self.snapshots = {
            "gpt-6-luna": "gpt-6-luna-2026-08-14", "gpt-6-sol": "gpt-6-sol-2026-07-30", "gpt-6-astra": "gpt-6-astra-2026-09-10",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18", "text-embedding-3-small": "text-embedding-3-small",
        }  # fmt: skip
        self.effort_tokens = dict(EFFORT_TOKENS)  # tests flatten this to make effort "ignored"
        self.reject_effort: set[str] = set()  # aliases that 400 on any reasoning effort
        self.no_reasoning_field = False
        self.compact_models = {"gpt-6-luna", "gpt-6-sol", "gpt-6-astra"}  # aliases with a /responses/compact route (else 404)
        self.compact_error: dict[str, tuple[int, dict]] = {}  # alias -> the compaction call's error reply
        self.reject_compaction_input: set[str] = set()  # aliases whose next call refuses a compaction item (400)
        self.requests: list[tuple[str, dict]] = []

    def compact(self, model: str, body: dict) -> tuple[int, dict]:
        if model in self.compact_error:
            return self.compact_error[model]
        if model not in self.compact_models:
            return 404, {"error": {"message": "Not found", "type": "invalid_request_error"}}
        return 200, {
            "id": "cmp_1", "object": "response.compaction", "created_at": 1_790_000_000,
            "output": [{"type": "compaction", "id": "cmpi_1", "encrypted_content": f"gAAAA-encrypted-{model}"}],
            "usage": {"input_tokens": 90, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 130},
        }  # fmt: skip

    def respond(self, path: str, body: dict) -> tuple[int, dict]:
        self.requests.append((path, body))
        model = body.get("model", "")
        served = self.snapshots.get(model, model)
        if path.endswith("/responses/compact"):
            return self.compact(model, body)
        compaction = any(isinstance(i, dict) and i.get("type") == "compaction" for i in (body.get("input") if isinstance(body.get("input"), list) else []))
        if compaction and model in self.reject_compaction_input:
            return 400, {"error": {"message": "Invalid input: items of type 'compaction' are not supported.", "type": "invalid_request_error", "param": "input", "code": None}}
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
        text = '{"ok": true}' if schema else probe_mod.NATIVE_CODE_WORD if compaction else "466"  # the block "remembers"
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


def _no_dotenv(monkeypatch) -> None:
    """The CLI loads the repository's .env; tests never read it (its keys would also leak into later tests)."""
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


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
    # Every Inspect model's API mode is recorded; without --study nothing else is added and no compaction is called.
    assert report["api_modes"] == {"openai/gpt-6-luna": {"api": "responses", "roles": ["agent", "kg", "judge"]}}
    assert not {"study", "native_compaction", "api_mode_problems"} & set(report)
    assert not any(p.endswith("/compact") for p, _ in fake_openai.requests)


def test_the_anchor_profile_uses_chat_completions_and_skips_effort_for_gpt_4o_mini(fake_openai, tmp_path):
    report = _probe(load_profile("anchor"), tmp_path)
    agent = report["roles"]["agent"]["paths"]["inspect"]
    assert agent["api"] == "chat.completions" and agent["effort"]["verdict"]["status"] == sc.SKIP
    assert report["api_modes"]["openai/gpt-4o-mini"]["api"] == "chat.completions" and report["api_modes"]["openai/gpt-6-luna"]["api"] == "responses"
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
    _no_dotenv(monkeypatch)
    out = tmp_path / "probe.json"
    probe_mod.main(["--profile", "gate", "--out", str(out)])
    report = json.loads(out.read_text())
    assert Ledger(tmp_path / "cache" / "ledger.jsonl").read(), "build-path probe calls are metered in the build ledger"
    assert report["version"] == sc.PROBE_VERSION and report["roles"]["agent"]["paths"]["inspect"]["api"] == "responses"
    printed = capsys.readouterr()
    assert "sk-fake-for-tests" not in printed.out + printed.err and '"check": "pass"' in printed.out


# ---------- Study G (BUILD_PLAN B11): every model of the study, API modes, native compaction ----------

G_MODELS = ("openai/gpt-6-luna", "openai/gpt-6-sol", "openai/gpt-6-astra")


def _study_g(fake, tmp_path: Path, record: Path | None) -> dict:
    base, *extra = [load_profile(n) for n in probe_mod.study_profiles("study_g")]
    return _probe(base, tmp_path, extra_profiles=extra, study="study_g", native_compaction=True, native_record=record)


def _compaction_followups(fake) -> list[str]:
    """The models of the calls that sent a compaction block back (the call after each compaction)."""
    return sorted(b["model"] for p, b in fake.requests if p.endswith("/responses") and isinstance(b.get("input"), list) and any(isinstance(i, dict) and i.get("type") == "compaction" for i in b["input"]))


def test_study_profiles_are_the_ones_the_study_cells_use():
    assert probe_mod.study_profiles("study_g") == ["study_g_luna", "study_g_sol", "study_g_astra"]
    assert probe_mod.study_profiles("main") == ["main_luna", "main_sol"]
    with pytest.raises(ValueError, match="unknown study 'nope'"):
        probe_mod.study_profiles("nope")


def test_the_study_g_probe_covers_every_model_records_api_modes_and_native_support(fake_openai, tmp_path, monkeypatch):
    from inspect_ai.model import GenerateConfig

    from ape.agent.cm_arms import NATIVE_RECORD_ENV, native_route

    record = tmp_path / "native_compaction.json"
    report = _study_g(fake_openai, tmp_path, record)
    roles = report["roles"]
    # Sol's and Astra's agents are probed like base roles; Luna's roles (the gate's settings) once.
    assert set(roles) == {"agent", "kg", "judge", "build", "agent@study_g_sol", "agent@study_g_astra"}
    shared = {"kg": "kg", "judge": "judge", "build": "build"}
    assert report["study"] == {
        "name": "study_g",
        "profiles": {"study_g_luna": {"agent": "agent"} | shared, "study_g_sol": {"agent": "agent@study_g_sol"} | shared, "study_g_astra": {"agent": "agent@study_g_astra"} | shared},
    }
    for name, snap in (("agent@study_g_sol", "gpt-6-sol-2026-07-30"), ("agent@study_g_astra", "gpt-6-astra-2026-09-10")):
        path = roles[name]["paths"]["inspect"]  # the honoured parameters: as configured, strict JSON, effort high > low
        assert roles[name]["profile"] == name.split("@")[1] and roles[name]["configured_effort"] == "high"
        assert path["api"] == "responses" and path["as_configured"]["accepted"] and path["structured_output"]["accepted"] and path["snapshot"] == snap
        assert path["effort"]["verdict"]["status"] == sc.PASS
    assert sc.effort_from_probe(report)["status"] == sc.PASS, "smoke's effort check covers the study's models"
    # One API mode per model, so per tier (every arm at a tier runs on the tier's agent model).
    assert {m: v["api"] for m, v in report["api_modes"].items()} == dict.fromkeys(G_MODELS, "responses") and "api_mode_problems" not in report
    assert report["api_modes"]["openai/gpt-6-astra"]["roles"] == ["agent@study_g_astra"]
    # Snapshots (for --pin and the live preflight) and rate limits for every alias.
    assert {a: report["snapshots"][a]["snapshot"] for a in ("gpt-6-sol", "gpt-6-astra")} == {"gpt-6-sol": "gpt-6-sol-2026-07-30", "gpt-6-astra": "gpt-6-astra-2026-09-10"}
    assert report["snapshots"]["gpt-6-astra"]["roles"] == ["agent@study_g_astra (inspect)"]
    assert {"gpt-6-luna", "gpt-6-sol", "gpt-6-astra"} <= set(report["ratelimits"])
    # Native compaction: one compaction and one call on its result per agent model, each confirmed.
    nc = report["native_compaction"]
    assert set(nc) == set(G_MODELS) and all(v["supported"] and v["outcome"] == "supported" and v["recalled"] and v["opaque_blocks"] == 1 for v in nc.values())
    assert nc["openai/gpt-6-sol"]["profile"] == "study_g_sol" and nc["openai/gpt-6-sol"]["compaction_usage"]["input_tokens"] == 90
    assert sorted(b["model"] for p, b in fake_openai.requests if p.endswith("/responses/compact")) == ["gpt-6-astra", "gpt-6-luna", "gpt-6-sol"]
    assert _compaction_followups(fake_openai) == ["gpt-6-astra", "gpt-6-luna", "gpt-6-sol"]
    # The support record, in cm_arms' format, read by CM-native's own gate under the names the runner and a session use.
    data = json.loads(record.read_text())
    assert data["format"] == 1 and set(data["models"]) == set(G_MODELS) and report["native_compaction_record"] == str(record)
    sol = data["models"]["openai/gpt-6-sol"]
    assert sol["supported"] is True and sol["outcome"] == "supported" and sol["api"] == "responses" and sol["snapshot"] == "gpt-6-sol-2026-07-30"
    assert sol["profile"] == "study_g_sol" and sol["source"] == "readiness/probe_openai.py" and "opaque compaction block" in sol["evidence"] and sol["recorded_at"]
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(record))
    assert [native_route(m) for m in G_MODELS] == ["provider"] * 3, "run_study's preflight passes the profile's model name"
    assert native_route(probe_mod.inspect_factory("openai/gpt-6-astra", GenerateConfig())) == "provider", "a session passes its Model"


def test_native_compaction_is_unsupported_without_the_endpoint_or_the_responses_api_and_when_its_block_is_refused(fake_openai, tmp_path, monkeypatch):
    from inspect_ai.model import GenerateConfig

    from ape.agent.cm_arms import NATIVE_RECORD_ENV, NativeCompactionUnsupported, native_route

    record = tmp_path / "native_compaction.json"
    fake_openai.compact_models = {"gpt-6-luna", "gpt-6-sol"}  # Astra: no compaction route
    fake_openai.reject_compaction_input = {"gpt-6-sol"}  # Sol compacts, but its next call refuses the block
    nc = _study_g(fake_openai, tmp_path, record)["native_compaction"]
    assert nc["openai/gpt-6-luna"]["outcome"] == "supported"
    assert nc["openai/gpt-6-sol"]["outcome"] == "rejected" and nc["openai/gpt-6-sol"]["supported"] is False and "rejected" in nc["openai/gpt-6-sol"]["evidence"]
    assert nc["openai/gpt-6-astra"]["outcome"] == "unsupported" and "not available" in nc["openai/gpt-6-astra"]["evidence"]
    assert _compaction_followups(fake_openai) == ["gpt-6-luna", "gpt-6-sol"], "no call after a compaction that failed"
    monkeypatch.setenv(NATIVE_RECORD_ENV, str(record))
    assert native_route("openai/gpt-6-luna") == "provider"
    for m in ("openai/gpt-6-sol", "openai/gpt-6-astra"):
        assert json.loads(record.read_text())["models"][m]["supported"] is False
        with pytest.raises(NativeCompactionUnsupported, match=m):
            native_route(m)
    # A model Inspect runs on chat completions (the anchor's gpt-4o-mini) has none, and no request is made for it.
    before = len(fake_openai.requests)
    res = asyncio.run(probe_mod.probe_native_compaction(probe_mod.inspect_factory("openai/gpt-4o-mini", GenerateConfig())))
    assert res["supported"] is False and res["outcome"] == "unsupported" and res["api"] == "chat.completions" and "Responses API" in res["evidence"]
    assert len(fake_openai.requests) == before


def test_errors_recorded_from_the_api_never_hold_the_key(fake_openai, tmp_path):
    key = "sk-fake-for-tests"  # the fixture's key, as the client sends it
    fake_openai.compact_error = {"gpt-6-luna": (401, {"error": {"message": f"Incorrect API key provided: {key}.", "type": "invalid_request_error", "code": "invalid_api_key"}})}
    record = tmp_path / "native_compaction.json"
    report = _study_g(fake_openai, tmp_path, record)
    luna = report["native_compaction"]["openai/gpt-6-luna"]
    assert luna["outcome"] == "error" and "Incorrect API key provided: sk-[redacted]" in luna["evidence"]
    assert key not in json.dumps(report) and key not in record.read_text()
    masked = probe_mod._error(RuntimeError("Incorrect API key provided: sk-proj-****abcd; also sk-live-123456789"))
    assert "abcd" not in masked and "123456789" not in masked and masked.startswith("RuntimeError: Incorrect API key provided: sk-[redacted]")


def test_the_dry_study_probe_runs_offline_and_writes_no_record(tmp_path, monkeypatch):
    """smoke.py --dry's path: mock models, no network. A mock has no native compaction; with no record path nothing is
    written, so a dry run can never confirm (or deny) a real model."""
    from types import SimpleNamespace

    from ape.agent.cm_arms import NATIVE_RECORD_ENV

    monkeypatch.setenv(NATIVE_RECORD_ENV, str(tmp_path / "native.json"))
    base, *extra = [load_profile(n) for n in probe_mod.study_profiles("study_g")]
    report = asyncio.run(probe_mod.run_probe(base, factory=probe_mod.offline_factory, build_client=probe_mod.OfflineBuildClient(), ledger=SimpleNamespace(append=lambda entry: None), extra_profiles=extra, study="study_g", native_compaction=True))
    assert set(report["roles"]) == {"agent", "kg", "judge", "build", "agent@study_g_sol", "agent@study_g_astra"} and sc.effort_from_probe(report)["status"] == sc.PASS
    assert {v["outcome"] for v in report["native_compaction"].values()} == {"unsupported"} and set(report["native_compaction"]) == set(G_MODELS)
    assert "native_compaction_record" not in report and not (tmp_path / "native.json").exists()


def test_cli_study_g_probe_writes_the_probe_and_the_support_record_and_never_prints_the_key(fake_openai, tmp_path, capsys, monkeypatch):
    from ape.agent.cm_arms import NATIVE_RECORD_ENV

    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))  # the build ledger and the default support record go here
    monkeypatch.delenv(NATIVE_RECORD_ENV, raising=False)
    _no_dotenv(monkeypatch)
    out = tmp_path / "probe.json"
    probe_mod.main(["--study", "study_g", "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["profile"] == "study_g_luna" and report["study"]["name"] == "study_g", "the study's own profile is the base"
    record = tmp_path / "cache" / "native_compaction.json"
    assert report["native_compaction_record"] == str(record) and set(json.loads(record.read_text())["models"]) == set(G_MODELS)
    printed = capsys.readouterr()
    assert "sk-fake-for-tests" not in printed.out + printed.err
    shown = json.loads(printed.out)
    assert shown["check"] == "pass" and shown["native_compaction"] == dict.fromkeys(G_MODELS, "supported") and shown["api_modes"] == dict.fromkeys(G_MODELS, "responses")
    # --native-record puts it elsewhere; an unknown study stops before any call.
    other = tmp_path / "elsewhere.json"
    probe_mod.main(["--study", "study_g", "--out", str(out), "--native-record", str(other)])
    assert set(json.loads(other.read_text())["models"]) == set(G_MODELS)
    capsys.readouterr()
    calls = len(fake_openai.requests)
    with pytest.raises(SystemExit):
        probe_mod.main(["--study", "nope", "--out", str(out)])
    assert len(fake_openai.requests) == calls and "unknown study 'nope'" in capsys.readouterr().err


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

    for k in [k for k in os.environ if k.startswith("APE_") and k != "APE_SPEND_REGISTRY"]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("APE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-never-sent")  # nothing here makes a request
    probe = tmp_path / "openai_probe.json"
    report = _report({"gpt-6-luna": ["s1"], "text-embedding-3-small": ["e1"]}, probed_at=f"{dt.date.today().isoformat()}T09:00:00+00:00")
    report["models_available"] = ["gpt-6-luna", "gpt-6-sol", "text-embedding-3-small"]
    probe.write_text(json.dumps(report))
    probe_mod.pin(probe, provenance)
    run = GateRun("live-pins", runs_root=tmp_path / "runs", env_path=tmp_path / "no.env", probe_path=probe, provenance_path=provenance, skip_smoke_check="test: the snapshot check alone")
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
