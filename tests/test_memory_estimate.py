"""Tests: the GGUF header reader, the memory estimate line by line (attention cache,
recurrent state of hybrid and state-space models, compute buffers, router mode), the machine's
GPU memory and the verdict, the Admin API (estimate, and the measured check as a job against a
stub engine), and the host helper's operations. Model files are written by the tests: nothing
depends on the models of this machine (the real ones are measured in the status comment).
"""

import json
import struct
import sys
from pathlib import Path

import pytest

from app.admin import memory_estimate as me
from app.admin.gguf import GgufError, read_header

MiB = 1024 * 1024


def _string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _value(value) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<IQ", 10, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if isinstance(value, list):
        return struct.pack("<IIQ", 9, 10, len(value)) + b"".join(
            struct.pack("<Q", v) for v in value
        )
    raise TypeError(value)


def write_gguf(path: Path, metadata: dict, tensor_bytes: int, version: int = 3) -> Path:
    """A GGUF file: the metadata, one tensor description, `tensor_bytes` of data."""
    head = b"GGUF" + struct.pack("<IQQ", version, 1, len(metadata))
    for key, value in metadata.items():
        head += _string(key) + _value(value)
    head += _string("w") + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<IQ", 0, 0)
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + b"\1" * tensor_bytes)
    return path


LLAMA = {
    "general.architecture": "llama", "llama.block_count": 32, "llama.context_length": 131072,
    "llama.embedding_length": 4096, "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8, "general.alignment": 32,
}  # fmt: skip
HYBRID = {
    "general.architecture": "qwen35", "qwen35.block_count": 64,
    "qwen35.context_length": 262144, "qwen35.embedding_length": 5120,
    "qwen35.attention.head_count": 24, "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256, "qwen35.attention.value_length": 256,
    "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.state_size": 128, "qwen35.ssm.group_count": 16,
    "qwen35.ssm.time_step_rank": 48, "qwen35.ssm.inner_size": 6144,
    "qwen35.full_attention_interval": 4,
}  # fmt: skip
MAMBA = {
    "general.architecture": "mamba", "mamba.block_count": 24, "mamba.embedding_length": 768,
    "mamba.ssm.conv_kernel": 4, "mamba.ssm.state_size": 16, "mamba.ssm.inner_size": 1536,
}  # fmt: skip


def _line(result, item):
    return next(line for line in result["lines"] if line["item"] == item)


# --- the reader ---


def test_the_header_gives_the_metadata_and_where_the_weights_start(tmp_path):
    path = write_gguf(tmp_path / "m.gguf", LLAMA, 5000)
    header = read_header(path)
    assert header["metadata"]["llama.block_count"] == 32
    assert header["tensors"] == 1
    assert header["file_size"] - header["data_offset"] == 5000
    assert header["data_offset"] % 32 == 0


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"SQLite format 3\0", "not a GGUF"),
        (b"GGUF" + struct.pack("<I", 1), "version 1"),
        (b"GGUF" + struct.pack("<IQQ", 3, 0, 5), "ends inside its header"),
    ],  # fmt: skip
)
def test_what_is_not_a_readable_gguf_is_refused(tmp_path, content, reason):
    path = tmp_path / "bad.gguf"
    path.write_bytes(content)
    with pytest.raises(GgufError, match=reason):
        read_header(path)


def test_a_long_array_is_skipped_not_kept(tmp_path):
    path = write_gguf(tmp_path / "v.gguf", {**LLAMA, "tokenizer.ggml.scores": list(range(5000))}, 8)
    header = read_header(path)
    assert header["metadata"]["tokenizer.ggml.scores"] is None
    assert header["file_size"] - header["data_offset"] == 8


# --- the estimate ---


def test_an_attention_model_counts_weights_and_the_shared_context_cache(tmp_path):
    path = write_gguf(tmp_path / "llama.gguf", LLAMA, 4096 * MiB // 1024)
    result = me.estimate(path, 16384, slots=4)
    kv = _line(result, "context cache")
    # 32 layers x 8 KV heads x (128 + 128) values x 16384 cells x 34/32 bytes = 1088 MiB
    assert kv["bytes"] == 32 * 8 * 256 * 16384 * 34 // 32 and kv["mib"] == 1088
    assert "4096 per slot, 4 slots share it" in kv["explanation"]
    assert _line(result, "weights")["bytes"] == 4 * MiB
    assert not any(line["item"] == "recurrent state" for line in result["lines"])
    assert result["total_bytes"] == sum(line["bytes"] for line in result["lines"])
    assert me.estimate(path, 65536)["lines"][1]["mib"] == 4352
    assert _line(me.estimate(path, 16384, cache_type="f16"), "context cache")["mib"] == 2048


def test_a_hybrid_model_counts_its_attention_layers_and_its_recurrent_state(tmp_path):
    path = write_gguf(tmp_path / "hybrid.gguf", HYBRID, 1024)
    result = me.estimate(path, 16384, slots=4)
    assert _line(result, "context cache")["mib"] == 544  # 16 attention layers of 64
    assert "16 attention layers of 64" in _line(result, "context cache")["explanation"]
    state = _line(result, "recurrent state")
    # 48 layers x (48 x 128 x 128 + 3 x 10240) float32 x 4 slots
    assert state["bytes"] == 48 * (48 * 128 * 128 + 3 * 10240) * 4 * 4
    assert state["mib"] == 598  # 598.5 MiB; the engine: 1142 - 544 = 598 MiB
    assert _line(me.estimate(path, 65536), "recurrent state")["bytes"] == state["bytes"]


def test_a_state_space_model_has_no_attention_cache(tmp_path):
    path = write_gguf(tmp_path / "mamba.gguf", MAMBA, 1024)
    result = me.estimate(path, 8192, slots=2)
    assert not any(line["item"] == "context cache" for line in result["lines"])
    assert _line(result, "recurrent state")["bytes"] == 24 * (1536 * 16 + 3 * 1536) * 4 * 2


def test_an_unknown_recurrent_state_is_said_not_hidden(tmp_path):
    odd = {k.replace("qwen35", "oddnet"): v for k, v in HYBRID.items() if "ssm" not in k}
    odd["general.architecture"] = "oddnet"
    odd["oddnet.ssm.conv_kernel"] = 4
    path = write_gguf(tmp_path / "odd.gguf", odd, 1024)
    notes = me.estimate(path, 4096)["notes"]
    assert any("recurrent state this estimate does not know" in n for n in notes)


def test_the_compute_buffers_are_an_upper_estimate_above_the_engines_figures():
    # The engine's own figures (MiB) for four models at 16384 and 65536 cells.
    measured = {16384: [37, 65, 170, 244], 65536: [70, 117, 374, 448]}
    for cells, values in measured.items():
        assert me.compute_estimate(cells) / MiB >= max(values)
        assert me.compute_estimate(cells) / MiB - max(values) < 300


def test_the_context_is_rounded_and_router_mode_and_training_length_are_noted(tmp_path):
    path = write_gguf(tmp_path / "llama.gguf", LLAMA, 1024)
    assert me.estimate(path, 1000)["ctx"] == 1024
    routed = me.estimate(path, 16384, router_models=3)
    assert (
        routed["router_total_mib"] == routed["total_mib"] * 3
        or abs(routed["router_total_mib"] - routed["total_mib"] * 3) <= 2
    )
    small = write_gguf(tmp_path / "short.gguf", {**LLAMA, "llama.context_length": 2048}, 1024)
    assert any("trained on" in n for n in me.estimate(small, 16384, slots=1)["notes"])


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"ctx": 10}, "ctx is"),
        ({"ctx": 4096, "slots": 0}, "slots is"),
        ({"ctx": 4096, "cache_type": "q9"}, "cache_type is"),
        ({"ctx": 4096, "router_models": 99}, "router_models is"),
    ],  # fmt: skip
)
def test_bad_settings_are_refused(tmp_path, kwargs, reason):
    path = write_gguf(tmp_path / "llama.gguf", LLAMA, 1024)
    ctx = kwargs.pop("ctx")
    with pytest.raises(me.EstimateError, match=reason):
        me.estimate(path, ctx, **kwargs)


# --- the machine and the verdict ---


def _fake_engine(tmp_path, devices: str) -> str:
    binary = tmp_path / "llama-server"
    binary.write_text(f"#!/bin/sh\nprintf '%s\\n' 'Available devices:' '{devices}'\n")
    binary.chmod(0o755)
    return str(binary)


def test_the_gpu_memory_is_read_from_the_engine_or_estimated(tmp_path, monkeypatch):
    binary = _fake_engine(tmp_path, "  MTL0: Apple M1 Pro (25559 MiB, 25558 MiB free)")
    info = me.machine(binary)
    assert (info["gpu_limit_mib"], info["gpu_limit_source"]) == (
        25559, "llama-server --list-devices"
    )  # fmt: skip
    monkeypatch.setattr(me, "physical_memory", lambda: 16 * 1024 * MiB)
    fallback = me.machine(str(tmp_path / "missing"))
    assert fallback["gpu_limit_mib"] == 12288 and "75%" in fallback["gpu_limit_source"]


@pytest.mark.parametrize(
    ("total", "fits"),
    [(8000, "yes"), (21725, "yes"), (21726, "tight"), (25559, "tight"), (25560, "no")],  # fmt: skip
)
def test_the_verdict_has_a_margin_below_the_gpu_limit(total, fits):
    answer = me.verdict(total, {"gpu_limit_mib": 25559})
    assert answer["fits"] == fits, answer
    if fits == "no":
        assert "1 MiB too many" in answer["reason"] and "smaller context" in answer["reason"]


def test_without_a_known_gpu_the_verdict_says_so():
    assert me.verdict(100, {"gpu_limit_mib": None})["fits"] == "unknown"


# --- the Admin API ---


ENGINE_STUB = r"""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def do_GET(self): self._send({"status": "ok"})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"choices": [{"message": {"content": "ok"}}],
                    "timings": {"prompt_n": 180, "prompt_per_second": 612.5,
                                "predicted_n": 64, "predicted_per_second": SPEED}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
async def api(fresh_db, monkeypatch, tmp_path):
    import secrets

    import httpx

    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "llama.gguf", LLAMA, 1024)
    key = secrets.token_hex(24)
    monkeypatch.setenv("API_SERVER_KEY", key)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("LLAMA_CTX_SIZE", "16384")
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    await init_db()

    def engine(speed=24.0, devices="  MTL0: Test GPU (25559 MiB, 25558 MiB free)"):
        """A stub llama-server: its devices, then a server with the given reply speed. Its code
        is in a file: the measured check starts the engine with a cleared environment."""
        (tmp_path / "bin").mkdir(exist_ok=True)
        stub = tmp_path / "bin" / "engine_stub.py"
        stub.write_text(ENGINE_STUB.replace("SPEED", str(speed)))
        binary = tmp_path / "bin" / "llama-server"
        binary.write_text(
            "#!/bin/sh\n"
            f"if [ \"$1\" = --list-devices ]; then printf '%s\\n' '{devices}'; exit 0; fi\n"
            f'exec {sys.executable} {stub} "$@"\n'
        )
        binary.chmod(0o755)
        monkeypatch.setenv("LLAMA_SERVER_BIN", str(binary))
        get_settings.cache_clear()

    engine()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        c.app, c.engine = app, engine
        yield c
    app.dependency_overrides.clear()
    deps.reset_failure_state()
    registry.clear()
    get_settings.cache_clear()


async def test_the_estimate_route_answers_the_lines_the_machine_and_the_verdict(api):
    answer = await api.get("/models/llama.gguf/estimate", params={"ctx": 16384, "slots": 4})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert [line["item"] for line in body["lines"]] == [
        "weights", "context cache", "compute buffers",
    ]  # fmt: skip
    assert body["machine"]["gpu_limit_mib"] == 25559 and body["verdict"]["fits"] == "yes"
    assert (await api.get("/models/nope.gguf/estimate")).status_code == 404
    assert (await api.get("/models/../x/estimate")).status_code in (404, 422)
    bad = await api.get("/models/llama.gguf/estimate", params={"cache_type": "q9"})
    assert bad.status_code == 422 and "cache_type is" in bad.json()["detail"]


async def _job(api, response):
    from app.admin.jobs import registry

    assert response.status_code == 202, response.text
    job = registry.get(response.json()["id"])
    await job.task
    return job


async def test_the_measured_check_reports_the_engines_speed(api, monkeypatch):
    monkeypatch.setattr(me, "swap_used_mib", lambda: 100.0)
    job = await _job(api, await api.post("/models/llama.gguf/benchmark", params={"ctx": 4096}))
    assert job.status == "done", job.error
    result = job.result
    assert (result["reply_tokens_per_second"], result["prompt_tokens_per_second"]) == (24.0, 612.5)
    assert result["smooth"] is True and result["swap_growth_mib"] == 0.0
    api.engine(speed=4.0)
    slow = await _job(api, await api.post("/models/llama.gguf/benchmark", params={"ctx": 4096}))
    assert slow.result["smooth"] is False and "4.0 tokens/s" in slow.result["reason"]


async def test_the_measured_check_is_refused_when_the_model_does_not_fit(api):
    api.engine(devices="  MTL0: Tiny GPU (100 MiB, 100 MiB free)")
    job = await _job(api, await api.post("/models/llama.gguf/benchmark", params={"ctx": 65536}))
    assert job.status == "failed" and job.error.startswith("not measured:")


async def test_the_routes_need_their_scopes(api):
    from app.api.scopes import Principal, Scope, get_principal

    api.app.dependency_overrides[get_principal] = lambda: Principal("op", Scope.OPERATE)
    assert (await api.get("/models/llama.gguf/estimate")).status_code == 200
    assert (await api.post("/models/llama.gguf/benchmark")).status_code == 403


async def test_the_host_helper_answers_the_estimate_with_the_same_function(api):
    from app.host import helper

    body = await helper._models_estimate({"name": "llama.gguf", "ctx": "16384"}, None)
    assert (
        body["total_mib"]
        == json.loads((await api.get("/models/llama.gguf/estimate", params={"ctx": 16384})).text)[
            "total_mib"
        ]
    )
    with pytest.raises(Exception, match="whole number"):
        await helper._models_estimate({"name": "llama.gguf", "ctx": "big"}, None)


@pytest.mark.parametrize(
    ("output", "used"),
    [("total = 2048.00M  used = 1203.25M  free = 844.75M  (encrypted)", 1203.25),
     ("total = 0,00M  used = 0,00M  free = 0,00M  (encrypted)", 0.0),
     ("vm.swapusage: unknown", None)],  # fmt: skip
)
def test_the_swap_is_read_whatever_the_decimal_mark(monkeypatch, output, used):
    """A French macOS writes "0,00M": the swap was read as unknown on the owner's Mac."""
    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=output)
    )
    assert me.swap_used_mib() == used
