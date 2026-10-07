"""How much memory a model needs with a given context, and whether it fits this machine.

The estimate is read from the model file itself (its GGUF metadata, `app/admin/gguf.py`), line
by line, each with a sentence that says what it is:

- weights: the tensor data of the file (the rest of the file after its header), plus the
  multimodal projector when one is given;
- context cache: for each attention layer, the keys and the values of every context cell,
  2 x layers x cells x KV heads x head size x bytes per element of the cache type. The engine
  shares one cache between its slots (`n_ctx` cells in all, `n_ctx / slots` each);
- recurrent state: for a hybrid or state-space model, the state each non-attention layer keeps
  per slot (float32), whatever the context;
- compute buffers: an upper estimate (a constant and a share per context cell), fitted on the
  engine's own figures, stated as such.

Measured against llama-server's memory breakdown for four models at 16384 and 65536 cells
(a probe of the real engine): weights and context within 1%.

The machine: the GPU memory the engine may use (`llama-server --list-devices`, on Apple Silicon
the Metal working-set limit; else 75% of the physical memory, stated), the physical memory, and
the verdict: fits with margin (at most 85% of the GPU limit), tight (at most the limit), does
not fit (with what to change).
"""

import math
import os
import re
import subprocess
from pathlib import Path

from app.admin.gguf import GgufError, read_header

MiB = 1024 * 1024
# Bytes per element of a cache type: a block of 32 values and its scale (ggml's block sizes).
CACHE_TYPES = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
    "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32,
}  # fmt: skip
DEFAULT_CACHE_TYPE = "q8_0"  # start.sh starts the engine with -ctk q8_0 -ctv q8_0
DEFAULT_SLOTS = 4  # llama-server's default number of parallel slots
CTX_PADDING = 256  # the engine rounds the context up to a multiple of 256 cells
COMPUTE_BASE = 200 * MiB  # upper estimate, see compute_estimate()
COMPUTE_PER_CELL = 4352
MARGIN_SHARE = 0.85
FALLBACK_GPU_SHARE = 0.75
MAX_CTX = 1 << 21


class EstimateError(Exception):
    """A refusal whose message is safe to show."""


def _mib(value: float) -> int:
    return int(round(value / MiB))


def _arch_value(metadata: dict, arch: str, key: str, default=None):
    return metadata.get(f"{arch}.{key}", default)


def attention_layers(metadata: dict, arch: str, blocks: int) -> list[int]:
    """The indexes of the layers with an attention cache. A hybrid model says one full-attention
    layer every `full_attention_interval` (the last of each group); a pure state-space model
    (mamba) has none."""
    interval = _arch_value(metadata, arch, "full_attention_interval")
    if interval:
        return [i for i in range(blocks) if (i + 1) % int(interval) == 0]
    if arch.startswith("mamba") or arch in ("rwkv6", "rwkv7"):
        return []
    return list(range(blocks))


def _per_layer(value, layer: int) -> int:
    return int(value[layer] if isinstance(value, list) else value)


def recurrent_state_bytes(metadata: dict, arch: str, layers: int) -> tuple[int, str] | None:
    """Bytes of recurrent state per slot for `layers` non-attention layers, float32, and how it
    was computed; None when the architecture's state is not known to this estimate."""
    if not layers:
        return 0, ""
    get = lambda key: _arch_value(metadata, arch, f"ssm.{key}")  # noqa: E731
    conv, state, groups, heads, inner = (
        get("conv_kernel"), get("state_size"), get("group_count"), get("time_step_rank"),
        get("inner_size"),
    )  # fmt: skip
    if None in (conv, state, inner):
        return None
    if groups and heads:  # gated delta net (qwen3-next, qwen3.5): a d_k x d_v matrix per head
        head_v = inner // heads
        matrix = heads * state * head_v
        conv_state = (conv - 1) * (2 * groups * state + inner)
        how = f"{heads} heads x {state} x {head_v} + {conv - 1} x {2 * groups * state + inner}"
    else:  # mamba: d_inner x d_state, and the convolution
        matrix = inner * state
        conv_state = (conv - 1) * inner
        how = f"{inner} x {state} + {conv - 1} x {inner}"
    return (matrix + conv_state) * 4 * layers, how


def compute_estimate(cells: int) -> int:
    """An upper estimate of the engine's compute buffers: 200 MiB plus 4352 bytes per context
    cell. Fitted above the engine's figures for four models (37 to 448 MiB measured)."""
    return COMPUTE_BASE + COMPUTE_PER_CELL * cells


def estimate(
    model: Path,
    ctx: int,
    *,
    slots: int = DEFAULT_SLOTS,
    cache_type: str = DEFAULT_CACHE_TYPE,
    projector: Path | None = None,
    router_models: int = 1,
) -> dict:
    """The memory `model` needs with `ctx` context cells, line by line (bytes and MiB, and what
    each line is), and the total."""
    if not isinstance(ctx, int) or not 256 <= ctx <= MAX_CTX:
        raise EstimateError(f"ctx is a number of tokens from 256 to {MAX_CTX}")
    if not isinstance(slots, int) or not 1 <= slots <= 64:
        raise EstimateError("slots is from 1 to 64")
    if cache_type not in CACHE_TYPES:
        raise EstimateError(f"cache_type is one of: {', '.join(CACHE_TYPES)}")
    if not isinstance(router_models, int) or not 1 <= router_models <= 16:
        raise EstimateError("router_models is from 1 to 16")
    try:
        header = read_header(model)
    except (GgufError, OSError) as exc:
        raise EstimateError(f"{model.name}: {exc}") from None
    metadata = header["metadata"]
    arch = str(metadata.get("general.architecture", ""))
    blocks = _arch_value(metadata, arch, "block_count")
    if not arch or not blocks:
        raise EstimateError(f"{model.name}: the file does not say its architecture or layers")
    blocks = int(blocks)
    cells = math.ceil(ctx / CTX_PADDING) * CTX_PADDING
    lines, notes = [], []

    weights = header["file_size"] - header["data_offset"]
    lines.append({
        "item": "weights",
        "bytes": weights,
        "explanation": f"The model's {header['tensors']} tensors as stored in the file, loaded "
        "whole into GPU memory (every layer on the GPU, -ngl 99).",
    })  # fmt: skip
    if projector is not None and projector.exists():
        try:
            proj = read_header(projector)
            lines.append({
                "item": "projector",
                "bytes": proj["file_size"] - proj["data_offset"],
                "explanation": f"The image projector {projector.name}, loaded beside the model.",
            })  # fmt: skip
        except GgufError:
            notes.append(f"{projector.name} is not a GGUF file: not counted")

    attention = attention_layers(metadata, arch, blocks)
    heads = _arch_value(metadata, arch, "attention.head_count")
    kv_heads = _arch_value(metadata, arch, "attention.head_count_kv", heads)
    embd = _arch_value(metadata, arch, "embedding_length")
    key_len = _arch_value(metadata, arch, "attention.key_length")
    value_len = _arch_value(metadata, arch, "attention.value_length")
    if attention and (kv_heads is None or (key_len is None and not (embd and heads))):
        raise EstimateError(f"{model.name}: the file does not give its attention sizes")
    if attention:
        head = embd // _per_layer(heads, 0) if embd and heads else None
        key_len, value_len = int(key_len or head), int(value_len or key_len or head)
        per_cell = sum(_per_layer(kv_heads, i) * (key_len + value_len) for i in attention)
        kv = per_cell * cells * CACHE_TYPES[cache_type]
        lines.append({
            "item": "context cache",
            "bytes": int(kv),
            "explanation": f"Keys and values of {cells} context cells ({cells // slots} per slot, "
            f"{slots} slots share it) in {len(attention)} attention layers of {blocks}: "
            f"{_per_layer(kv_heads, attention[0])} KV heads x ({key_len} + {value_len}) values "
            f"x {CACHE_TYPES[cache_type]:.4g} bytes ({cache_type}).",
        })  # fmt: skip
    recurrent_layers = blocks - len(attention)
    state = recurrent_state_bytes(metadata, arch, recurrent_layers)
    if state is None:
        notes.append(
            f"{recurrent_layers} layers of {arch} keep a recurrent state this estimate does not "
            "know how to size: the total is too low by that much"
        )
    elif state[0]:
        lines.append({
            "item": "recurrent state",
            "bytes": state[0] * slots,
            "explanation": f"The state each of the {recurrent_layers} non-attention layers keeps "
            f"per slot, whatever the context: {state[1]} values, float32, x {slots} slots.",
        })  # fmt: skip
    lines.append({
        "item": "compute buffers",
        "bytes": compute_estimate(cells),
        "explanation": "The engine's working memory while it computes (an upper estimate: 200 "
        "MiB plus 4352 bytes per context cell, above what the engine reported for four models).",
    })  # fmt: skip
    for line in lines:
        line["mib"] = _mib(line["bytes"])
    total = sum(line["bytes"] for line in lines)
    trained = _arch_value(metadata, arch, "context_length")
    if trained and cells // slots > int(trained):
        notes.append(
            f"each slot gets {cells // slots} cells, more than the {int(trained)} the model was "
            "trained on: the replies past that length degrade"
        )
    result = {
        "model": model.name,
        "architecture": arch,
        "ctx": cells,
        "slots": slots,
        "cache_type": cache_type,
        "lines": lines,
        "total_bytes": total,
        "total_mib": _mib(total),
        "notes": notes,
    }
    if router_models > 1:
        result["router_models"] = router_models
        result["router_total_mib"] = _mib(total * router_models)
        notes.append(
            f"router mode: up to {router_models} models loaded at once; the router total counts "
            f"{router_models} models of this size"
        )
    return result


# --- the machine ---


_DEVICE = re.compile(r"^\s*(\w+): (.+?) \((\d+) MiB, (\d+) MiB free\)", re.M)


def gpu_devices(llama_server: str | None) -> list[dict]:
    """The engine's devices and their memory, as `llama-server --list-devices` prints them
    (on Apple Silicon: the Metal working-set limit). Empty when it cannot be asked."""
    if not llama_server or not Path(llama_server).is_file():
        return []
    try:
        out = subprocess.run(
            [llama_server, "--list-devices"], capture_output=True, text=True, timeout=30,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [
        {"device": m.group(1), "name": m.group(2), "total_mib": int(m.group(3))}
        for m in _DEVICE.finditer(out.stdout + out.stderr)
        if int(m.group(3)) > 0
    ]


def physical_memory() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def machine(llama_server: str | None) -> dict:
    physical = physical_memory()
    devices = gpu_devices(llama_server)
    gpu = max((d["total_mib"] for d in devices), default=None)
    source = "llama-server --list-devices"
    if gpu is None and physical:
        gpu, source = int(physical * FALLBACK_GPU_SHARE / MiB), "75% of the physical memory"
    return {
        "physical_mib": _mib(physical) if physical else None,
        "gpu_limit_mib": gpu,
        "gpu_limit_source": source if gpu else None,
        "devices": devices,
    }


def verdict(total_mib: int, machine_info: dict) -> dict:
    limit = machine_info.get("gpu_limit_mib")
    if not limit:
        return {"fits": "unknown", "reason": "the GPU memory of this machine is not known"}
    share = total_mib / limit
    advice = "a smaller context (ctx), fewer slots, a smaller cache type (q4_0) or quantization"
    if share <= MARGIN_SHARE:
        fits = "yes"
        reason = f"{total_mib} MiB is {share:.0%} of the {limit} MiB the GPU may use"
    elif share <= 1:
        fits = "tight"
        reason = (
            f"{total_mib} MiB is {share:.0%} of the {limit} MiB the GPU may use: little room "
            f"for anything else; the engine may shrink the context to fit. Consider {advice}"
        )
    else:
        fits = "no"
        reason = (
            f"{total_mib} MiB is more than the {limit} MiB the GPU may use "
            f"({total_mib - limit} MiB too many). Use {advice}"
        )
    return {"fits": fits, "share": round(share, 3), "reason": reason}


# --- the whole answer for one model of the models directory ---


def for_model(
    name: str,
    ctx: int | None = None,
    *,
    slots: int = DEFAULT_SLOTS,
    cache_type: str = DEFAULT_CACHE_TYPE,
    router_models: int = 1,
    projector: str | None = None,
) -> dict:
    """The estimate of `name` (a file of the models directory), this machine, the verdict."""
    from app.admin import models
    from app.admin.service import InvalidInputError, NotFoundError
    from app.config import get_settings

    settings = get_settings()
    path = models.model_path(name)
    if not path.is_file():
        raise NotFoundError(f"No model {name}")
    extra = None
    if projector:
        extra = models.model_path(projector)
        if not extra.is_file():
            raise NotFoundError(f"No model {projector}")
    try:
        result = estimate(
            path, ctx or settings.llama_ctx_size, slots=slots, cache_type=cache_type,
            projector=extra, router_models=router_models,
        )  # fmt: skip
    except EstimateError as exc:
        raise InvalidInputError(str(exc)) from None
    result["machine"] = machine(settings.llama_server_bin)
    total = result.get("router_total_mib", result["total_mib"])
    result["verdict"] = verdict(total, result["machine"])
    return result


# --- the measured check: does it run smoothly? ---

SMOOTH_TOKENS_PER_SECOND = 10.0  # a reply is read comfortably from about this speed
SWAP_GROWTH_MIB = 64  # more swap than this during the run means the machine ran short
BENCH_PROMPT = (
    "Here is a short note about sea otters. Sea otters live along the coasts of the northern "
    "Pacific Ocean. They hold hands while they sleep so that they do not drift apart, they eat "
    "sea urchins, crabs and clams, and an adult has the densest fur of any animal. "
    * 4
    + "Summarise this note in three sentences."
)
_benchmark_lock = None


def swap_used_mib() -> float | None:
    """macOS: the swap in use (`sysctl vm.swapusage`); None elsewhere."""
    try:
        # /usr/sbin is not on every PATH (a cleared environment, a service): named in full.
        program = "/usr/sbin/sysctl" if Path("/usr/sbin/sysctl").exists() else "sysctl"
        out = subprocess.run(
            [program, "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    # The decimal mark follows the locale: "used = 0.00M" or "used = 0,00M".
    match = re.search(r"used = ([\d.,]+)M", out)
    return float(match.group(1).replace(",", ".")) if match else None


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def benchmark(job, name: str, ctx: int | None = None, slots: int = DEFAULT_SLOTS) -> dict:
    """Start the model in a temporary engine on a free local port with start.sh's flags, ask
    one question, read the engine's own speeds (tokens per second, prompt and reply) and the
    swap before and after; stop it. Refused when the estimate says it does not fit."""
    import asyncio
    import tempfile
    import time

    import httpx

    from app.admin.jobs import JobError
    from app.config import get_settings

    global _benchmark_lock
    _benchmark_lock = _benchmark_lock or asyncio.Lock()
    if _benchmark_lock.locked():
        raise JobError("another measurement is running")
    async with _benchmark_lock:
        settings = get_settings()
        planned = await asyncio.to_thread(for_model, name, ctx, slots=slots)
        if planned["verdict"]["fits"] == "no":
            raise JobError("not measured: " + planned["verdict"]["reason"])
        binary = settings.llama_server_bin
        if not binary or not Path(binary).is_file():
            raise JobError("LLAMA_SERVER_BIN does not name the llama-server program")
        port = _free_port()
        log = Path(tempfile.mkdtemp(prefix="benchmark_")) / "llama.log"
        swap_before = await asyncio.to_thread(swap_used_mib)
        started = time.monotonic()
        job.update(0.1, "Loading the model in a temporary engine")
        process = await asyncio.create_subprocess_exec(
            binary, "--port", str(port), "--host", "127.0.0.1",
            "--model", str(Path(settings.models_dir) / name), "--ctx-size", str(planned["ctx"]),
            "-np", str(slots), "-ngl", "99", "--jinja", "--flash-attn", "on",
            "-ctk", DEFAULT_CACHE_TYPE, "-ctv", DEFAULT_CACHE_TYPE,
            stdout=log.open("w"), stderr=asyncio.subprocess.STDOUT,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "HOME": os.environ.get("HOME", "/")},
        )  # fmt: skip
        base = f"http://127.0.0.1:{port}"
        try:
            async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
                for _ in range(600):
                    if process.returncode is not None:
                        raise JobError("the temporary engine stopped while loading the model")
                    try:
                        if (await client.get(f"{base}/health")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    raise JobError("the model did not load within 300 s")
                loaded = time.monotonic() - started
                job.update(0.6, "Asking one question")
                response = await client.post(
                    f"{base}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": BENCH_PROMPT}],
                        "max_tokens": 128,
                        "temperature": 0,
                    },  # fmt: skip
                    timeout=600,
                )
                response.raise_for_status()
                timings = response.json().get("timings") or {}
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 30)
                except TimeoutError:
                    process.kill()
        swap_after = await asyncio.to_thread(swap_used_mib)
        reply = timings.get("predicted_per_second")
        swap_growth = (
            round(swap_after - swap_before, 1) if None not in (swap_before, swap_after) else None
        )
        smooth = bool(reply and reply >= SMOOTH_TOKENS_PER_SECOND) and (
            swap_growth is None or swap_growth <= SWAP_GROWTH_MIB
        )
        return {
            "model": name,
            "ctx": planned["ctx"],
            "slots": slots,
            "load_seconds": round(loaded, 1),
            "prompt_tokens": timings.get("prompt_n"),
            "prompt_tokens_per_second": round(timings.get("prompt_per_second") or 0, 1),
            "reply_tokens": timings.get("predicted_n"),
            "reply_tokens_per_second": round(reply or 0, 1),
            "swap_growth_mib": swap_growth,
            "smooth": smooth,
            "reason": (
                f"replies at {reply or 0:.1f} tokens/s (smooth from "
                f"{SMOOTH_TOKENS_PER_SECOND:.0f}), swap grew {swap_growth} MiB (limit "
                f"{SWAP_GROWTH_MIB})"
            ),
            "estimate_total_mib": planned["total_mib"],
            "verdict": planned["verdict"],
        }
