"""Tests: pulling a model from a hub or a URL, as a resumable job, through the
outbound guard. A real HTTP server plays the hub; a mock transport plays the internet where
the guard's refusals and the token's path are checked.
"""

import asyncio
import dataclasses
import hashlib
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from app.security.outbound import Guard, guard_from_settings

KEY = "Zq8vT3mK9xW2pL7nR4bY6cH1dF5gJ0sA"
GOOD = {"Authorization": f"Bearer {KEY}"}
TOKEN = "hf_TheTokenThatMustNeverBeLogged_1234567890"
PUBLIC = "93.184.216.34"


def _data(size=2 * 1024 * 1024, seed=3):
    return b"GGUF" + bytes((seed * i) % 251 for i in range(4096)) * (size // 4096)


class Hub:
    """A hub on 127.0.0.1: /api/models/org/model, /org/model/resolve/main/<file> (a redirect
    that states the SHA256 like the real one), and /cdn/<file> with Range support."""

    def __init__(self, files, etag=None):
        self.files, self.etag = files, etag
        self.cut_after = None  # serve only this many bytes, then drop the connection
        self.delay = 0.0  # seconds between chunks
        self.requests = []
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hub.requests.append((self.path, dict(self.headers)))
                if self.path == "/api/models/org/model":
                    body = json.dumps({"siblings": [{"rfilename": n} for n in hub.files]}).encode()
                    return self._send(200, body, {"Content-Type": "application/json"})
                if self.path.startswith("/org/model/resolve/main/"):
                    name = self.path.rsplit("/", 1)[1]
                    data = hub.files.get(name)
                    if data is None:
                        return self._send(404, b"")
                    etag = hub.etag or hashlib.sha256(data).hexdigest()
                    return self._send(
                        302,
                        b"",
                        {
                            "Location": f"/cdn/{name}",
                            "X-Linked-ETag": f'"{etag}"',
                            "X-Linked-Size": str(len(data)),
                        },
                    )
                if self.path.startswith("/cdn/"):
                    return self._cdn(self.path.rsplit("/", 1)[1])
                self._send(404, b"")

            def _send(self, code, body, headers=()):
                self.send_response(code)
                for k, v in dict(headers).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _cdn(self, name):
                data = hub.files[name]
                start = 0
                rng = self.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].split("-")[0])
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                sent = 0
                while start + sent < len(data):
                    if hub.cut_after is not None and sent >= hub.cut_after:
                        self.wfile.flush()
                        self.connection.shutdown(2)
                        return
                    piece = data[start + sent : start + sent + 65536]
                    if hub.cut_after is not None:
                        piece = piece[: hub.cut_after - sent]
                    self.wfile.write(piece)
                    sent += len(piece)
                    if hub.delay:
                        time.sleep(hub.delay)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
async def client(fresh_db, monkeypatch, tmp_path):
    from app.admin import models
    from app.admin.jobs import registry
    from app.api import deps
    from app.api.app import app
    from app.config import get_settings
    from app.db.session import init_db

    directory = tmp_path / "models"
    directory.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(directory))
    monkeypatch.setenv("MODEL_FILE", "")
    monkeypatch.setenv("API_SERVER_KEY", KEY)
    get_settings.cache_clear()
    deps.reset_failure_state()
    registry.clear()
    models._active.clear()
    await init_db()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=GOOD) as c:
        c.dir, c.env = directory, monkeypatch
        yield c
    registry.clear()
    models._active.clear()
    deps.reset_failure_state()


@pytest.fixture
def hub(client, monkeypatch):
    """A local hub, and a guard built to allow it (http and a loopback address)."""
    from app.admin import models
    from app.config import get_settings

    files = {"m-q4_k_m.gguf": _data(seed=3), "m-q8_0.gguf": _data(seed=5)}
    h = Hub(files)
    monkeypatch.setenv("MODEL_HUB_URL", h.url)
    get_settings.cache_clear()
    guard = Guard(allowed_hosts=frozenset({"127.0.0.1"}), allow_http=True, allow_private=True)
    monkeypatch.setattr(models, "build_guard", lambda: guard)
    yield h
    h.close()


async def _wait(client, job_id, tries=400):
    for _ in range(tries):
        job = (await client.get(f"/jobs/{job_id}")).json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        await asyncio.sleep(0.03)
    raise AssertionError(job)


async def _pull(client, **body):
    return await client.post("/models/pull", json=body)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# --- from the hub ---


async def test_a_pull_by_repo_and_quant_downloads_checks_and_records(client, hub, caplog):
    with caplog.at_level(logging.DEBUG):
        started = await _pull(client, spec="org/model:q4_k_m")
        assert started.status_code == 202
        job = await _wait(client, started.json()["id"])
    data = hub.files["m-q4_k_m.gguf"]
    assert job["status"] == "done" and job["progress"] == 1.0
    assert job["result"]["sha256"] == _sha(data) and job["result"]["verified"] is True
    assert job["result"]["size_bytes"] == len(data) and job["result"]["name"] == "m-q4_k_m.gguf"
    assert (client.dir / "m-q4_k_m.gguf").read_bytes() == data
    assert (client.dir / "m-q4_k_m.gguf.sha256").read_text().split()[0] == _sha(data)
    assert sorted(p.name for p in client.dir.iterdir()) == ["m-q4_k_m.gguf", "m-q4_k_m.gguf.sha256"]
    events = (await client.get("/admin-events", params={"action": "model.pull"})).json()
    assert len(events) == 1 and _sha(data) in str(events[0]["details"])


async def test_a_wrong_hash_from_the_hub_refuses_the_file_and_leaves_nothing(client, hub):
    hub.etag = "0" * 64
    job = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert job["status"] == "failed" and "does not match" in job["error"]
    assert list(client.dir.iterdir()) == []


async def test_a_hash_given_by_the_administrator_is_checked_too(client, hub):
    good = _sha(hub.files["m-q4_k_m.gguf"])
    bad = await _wait(
        client, (await _pull(client, spec="org/model:q4_k_m", sha256="f" * 64)).json()["id"]
    )
    assert bad["status"] == "failed" and list(client.dir.iterdir()) == []
    ok = await _wait(
        client, (await _pull(client, spec="org/model:q4_k_m", sha256=good.upper())).json()["id"]
    )
    assert ok["status"] == "done"


async def test_an_ambiguous_or_missing_quantization_is_refused_with_the_candidates(client, hub):
    ambiguous = await _wait(client, (await _pull(client, spec="org/model")).json()["id"])
    assert ambiguous["status"] == "failed"
    assert "m-q4_k_m.gguf" in ambiguous["error"] and "m-q8_0.gguf" in ambiguous["error"]
    missing = await _wait(client, (await _pull(client, spec="org/model:q2_k")).json()["id"])
    assert missing["status"] == "failed" and "no .gguf file for q2_k" in missing["error"]
    assert list(client.dir.iterdir()) == []


async def test_sharded_models_and_projector_files_are_not_pulled(client, hub):
    hub.files.clear()
    hub.files.update({"big-q4-00001-of-00002.gguf": _data(), "mmproj-model-f16.gguf": _data()})
    shard = await _wait(client, (await _pull(client, spec="org/model:q4")).json()["id"])
    assert shard["status"] == "failed" and "sharded" in shard["error"]
    projector = await _wait(client, (await _pull(client, spec="org/model:f16")).json()["id"])
    assert projector["status"] == "failed" and "no .gguf file" in projector["error"]


async def test_an_interrupted_pull_resumes_and_ends_with_the_same_hash(client, hub):
    data = hub.files["m-q4_k_m.gguf"]
    hub.cut_after = len(data) // 2
    first = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    part = client.dir / "m-q4_k_m.gguf.part"
    assert first["status"] == "failed" and "resume" in first["error"]
    saved = part.stat().st_size  # whole blocks only: the tail of a dropped block is not kept
    assert 0 < saved <= len(data) // 2 and not (client.dir / "m-q4_k_m.gguf").exists()
    hub.cut_after = None
    hub.requests.clear()
    second = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert second["status"] == "done" and second["result"]["sha256"] == _sha(data)
    ranged = [h.get("Range") for p, h in hub.requests if p.startswith("/cdn/")]
    print(f"interrupted after {saved} of {len(data)} bytes kept, resumed with Range {ranged}")
    assert ranged == [f"bytes={saved}-"]
    assert (client.dir / "m-q4_k_m.gguf").read_bytes() == data and not part.exists()


async def test_a_cancelled_pull_keeps_its_partial_file_so_it_can_resume(client, hub):
    hub.delay = 0.02
    started = (await _pull(client, spec="org/model:q4_k_m")).json()
    for _ in range(200):
        if (client.dir / "m-q4_k_m.gguf.part").exists() and (
            client.dir / "m-q4_k_m.gguf.part"
        ).stat().st_size > 0:
            break
        await asyncio.sleep(0.02)
    assert (await client.post(f"/jobs/{started['id']}/cancel")).status_code == 200
    assert (await _wait(client, started["id"]))["status"] == "cancelled"
    assert not (client.dir / "m-q4_k_m.gguf").exists()
    hub.delay = 0.0
    again = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert again["status"] == "done"


async def test_a_size_over_the_limit_is_refused_with_no_bytes_written(client, hub):
    from app.config import get_settings

    client.env.setenv("MODEL_PULL_MAX_BYTES", "1000")
    get_settings.cache_clear()
    job = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert job["status"] == "failed" and "MODEL_PULL_MAX_BYTES" in job["error"]
    assert list(client.dir.iterdir()) == []


async def test_a_pull_past_the_time_limit_is_cut_and_can_be_resumed(client, hub):
    from app.config import get_settings

    client.env.setenv("MODEL_PULL_TIMEOUT_SECONDS", "1")
    get_settings.cache_clear()
    hub.delay = 0.1
    job = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert job["status"] == "failed" and "time limit" in job["error"]
    assert not (client.dir / "m-q4_k_m.gguf").exists()


async def test_a_name_that_exists_is_refused_unless_forced_and_two_pulls_of_a_name_conflict(
    client, hub
):
    (client.dir / "mine.gguf").write_bytes(b"GGUF")
    refused = await _pull(client, spec="org/model:q4_k_m", name="mine.gguf")
    assert refused.status_code == 409
    forced = await _pull(client, spec="org/model:q4_k_m", name="mine.gguf", force=True)
    assert (await _wait(client, forced.json()["id"]))["status"] == "done"
    hub.delay = 0.05
    first = await _pull(client, spec="org/model:q8_0", name="busy.gguf")
    await asyncio.sleep(0.2)
    second = await _pull(client, spec="org/model:q8_0", name="busy.gguf", force=True)
    third = await _wait(client, second.json()["id"])
    assert third["status"] == "failed" and "already being" in third["error"]
    await client.post(f"/jobs/{first.json()['id']}/cancel")


@pytest.mark.parametrize("spec", ["", "nonsense", "a/b/c", "org/model:", "ftp://x/y.gguf", "../x"])
async def test_a_bad_spec_is_refused_with_422(client, hub, spec):
    assert (await _pull(client, spec=spec)).status_code == 422


# --- from a URL ---


async def test_a_url_pull_takes_its_name_from_the_url(client, hub):
    data = hub.files["m-q8_0.gguf"]
    started = await _pull(client, spec=f"{hub.url}/cdn/m-q8_0.gguf")
    job = await _wait(client, started.json()["id"])
    assert job["status"] == "done" and job["result"]["name"] == "m-q8_0.gguf"
    assert job["result"]["sha256"] == _sha(data) and job["result"]["verified"] is False


async def test_a_url_without_a_model_name_needs_one(client, hub):
    assert (await _pull(client, spec=f"{hub.url}/cdn/")).status_code == 422
    ok = await _pull(client, spec=f"{hub.url}/cdn/m-q8_0.gguf", name="chosen.gguf")
    assert (await _wait(client, ok.json()["id"]))["result"]["name"] == "chosen.gguf"
    assert (
        await _pull(client, spec=f"{hub.url}/cdn/m-q8_0.gguf", name="a;b.gguf")
    ).status_code == 422


# --- the guard, with the production rules and a transport that fails the test if reached ---


class Internet:
    """A mock transport: hosts map to responses; every request is recorded."""

    def __init__(self, routes=None, body=None):
        self.routes, self.requests, self.body = routes or {}, [], body or _data()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.headers["host"].split(":")[0]
        route = self.routes.get(host)
        if route is None:
            return httpx.Response(
                200, content=self.body, headers={"content-length": str(len(self.body))}
            )
        return route(request)


@pytest.fixture
def internet(client, monkeypatch):
    """The production guard (https, allow-list, public addresses only) with DNS faked: every
    name resolves to a public address unless a test says otherwise."""
    from app.admin import models
    from app.config import get_settings

    net = Internet()
    net.dns = {}
    monkeypatch.delenv("MODEL_HUB_URL", raising=False)
    monkeypatch.delenv("MODEL_PULL_ALLOWED_HOSTS", raising=False)
    get_settings.cache_clear()

    def guard():
        base = guard_from_settings()
        return dataclasses.replace(base, resolver=lambda host, port: [net.dns.get(host, PUBLIC)])

    monkeypatch.setattr(models, "build_guard", guard)
    monkeypatch.setattr(
        models,
        "build_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(net.handler)),
    )
    return net


@pytest.mark.parametrize(
    "spec, fragment",
    [
        ("http://huggingface.co/o/m/resolve/main/m.gguf", "https"),
        ("https://127.0.0.1/m.gguf", "not allowed"),
        ("https://169.254.169.254/latest/m.gguf", "not allowed"),
        ("https://[::1]/m.gguf", "not allowed"),
        ("https://evil.example/m.gguf", "allowed hosts: hf.co, huggingface.co"),
        ("https://user:pw@huggingface.co/m.gguf", "credentials"),
    ],
)
async def test_a_url_the_guard_refuses_is_refused_at_once_with_nothing_written(
    client, internet, spec, fragment
):
    response = await _pull(client, spec=spec)
    assert response.status_code == 422 and fragment in response.json()["detail"]
    assert internet.requests == [] and list(client.dir.iterdir()) == []


async def test_an_allowed_name_that_resolves_inside_is_refused_before_anything_is_fetched(
    client, internet
):
    internet.dns["huggingface.co"] = "10.0.0.5"
    response = await _pull(client, spec="https://huggingface.co/o/m/resolve/main/m.gguf")
    assert response.status_code == 422 and "non-public" in response.json()["detail"]
    assert internet.requests == [] and list(client.dir.iterdir()) == []


async def test_a_public_url_that_redirects_to_a_private_address_is_refused_with_nothing_written(
    client, internet
):
    def redirect(request):
        return httpx.Response(302, headers={"location": "https://10.0.0.5/m.gguf"})

    internet.routes["huggingface.co"] = redirect
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "not allowed" in job["error"]
    assert len(internet.requests) == 1 and list(client.dir.iterdir()) == []


async def test_a_redirect_to_an_allowed_name_that_resolves_inside_is_refused_too(client, internet):
    internet.dns["cdn.huggingface.co"] = "192.168.1.9"

    def redirect(request):
        return httpx.Response(302, headers={"location": "https://cdn.huggingface.co/m.gguf"})

    internet.routes["huggingface.co"] = redirect
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "non-public" in job["error"]
    assert len(internet.requests) == 1 and list(client.dir.iterdir()) == []


async def test_adding_the_host_to_the_allow_list_accepts_the_url(client, internet):
    refused = await _pull(client, spec="https://models.example.net/m.gguf")
    assert refused.status_code == 422 and "allowed hosts" in refused.json()["detail"]
    from app.config import get_settings

    client.env.setenv("MODEL_PULL_ALLOWED_HOSTS", "models.example.net")
    get_settings.cache_clear()
    job = await _wait(
        client, (await _pull(client, spec="https://models.example.net/m.gguf")).json()["id"]
    )
    assert job["status"] == "done" and (client.dir / "m.gguf").exists()


async def test_the_request_goes_to_the_checked_address_with_the_original_name(client, internet):
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "done"
    request = internet.requests[0]
    assert request.url.host == PUBLIC  # the address that was checked, not the name
    assert request.headers["host"] == "huggingface.co"
    assert (
        request.extensions["sni_hostname"] == b"huggingface.co"
        or request.extensions["sni_hostname"] == "huggingface.co"
    )


# --- the token ---


async def test_the_token_goes_to_the_hub_only_and_is_never_logged_or_returned(
    client, internet, caplog
):
    from app.config import get_settings

    client.env.setenv("HF_TOKEN", TOKEN)
    get_settings.cache_clear()
    body = _data()
    api = json.dumps({"siblings": [{"rfilename": "m-q4_k_m.gguf"}]}).encode()

    def hub(request):
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, content=api)
        return httpx.Response(
            302,
            headers={
                "location": "https://cdn-lfs.huggingface.co/m",
                "x-linked-etag": f'"{_sha(body)}"',
            },
        )

    def cdn(request):
        return httpx.Response(200, content=body, headers={"content-length": str(len(body))})

    internet.routes.update({"huggingface.co": hub, "cdn-lfs.huggingface.co": cdn})
    with caplog.at_level(logging.DEBUG):
        started = await _pull(client, spec="org/model:q4_k_m")
        job = await _wait(client, started.json()["id"])
    assert job["status"] == "done" and job["result"]["verified"] is True
    by_host = {}
    for r in internet.requests:
        by_host.setdefault(r.headers["host"], []).append(r.headers.get("authorization"))
    print(f"authorization by host: { {h: [bool(a) for a in v] for h, v in by_host.items()} }")
    assert all(a == f"Bearer {TOKEN}" for a in by_host["huggingface.co"])
    assert by_host["cdn-lfs.huggingface.co"] == [None]
    everything = json.dumps(job) + started.text + "\n".join(r.getMessage() for r in caplog.records)
    everything += str((await client.get("/admin-events")).json()) + str(
        (await client.get("/jobs")).json()
    )
    assert TOKEN not in everything


async def test_the_token_is_not_sent_to_a_url_host_that_is_not_the_hub(client, internet):
    from app.config import get_settings

    client.env.setenv("HF_TOKEN", TOKEN)
    client.env.setenv("MODEL_PULL_ALLOWED_HOSTS", "models.example.net")
    get_settings.cache_clear()
    job = await _wait(
        client, (await _pull(client, spec="https://models.example.net/m.gguf")).json()["id"]
    )
    assert job["status"] == "done"
    assert [r.headers.get("authorization") for r in internet.requests] == [None]


async def test_without_a_models_directory_the_answer_is_409(client, internet):
    from app.config import get_settings

    client.env.setenv("MODELS_DIR", str(client.dir / "missing"))
    get_settings.cache_clear()
    assert (await _pull(client, spec="https://huggingface.co/o/m.gguf")).status_code == 409


# --- the download's own checks, with a mock transport ---


def _serve(net, host, handler):
    net.routes[host] = handler


async def test_a_range_answer_that_does_not_continue_the_file_is_refused(client, internet):
    (client.dir / "m.gguf.part").write_bytes(b"GGUF" + b"x" * 96)

    def wrong_range(request):
        return httpx.Response(206, content=b"y" * 10, headers={"content-range": "bytes 0-9/100"})

    _serve(internet, "huggingface.co", wrong_range)
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "does not continue" in job["error"]
    assert not (client.dir / "m.gguf").exists()


async def test_the_size_limit_holds_when_the_server_does_not_announce_a_size(client, internet):
    from app.config import get_settings

    client.env.setenv("MODEL_PULL_MAX_BYTES", "500000")
    get_settings.cache_clear()
    body = _data()

    async def unannounced():
        for i in range(0, len(body), 65536):
            yield body[i : i + 65536]

    _serve(internet, "huggingface.co", lambda request: httpx.Response(200, content=unannounced()))
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "MODEL_PULL_MAX_BYTES" in job["error"]
    assert list(client.dir.iterdir()) == []


async def test_not_enough_free_space_is_refused_before_writing(client, internet, monkeypatch):
    from app.admin import models

    monkeypatch.setattr(models, "free_bytes", lambda _d: 1024)
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "free disk space" in job["error"]
    assert list(client.dir.iterdir()) == []


async def test_a_stream_that_ends_early_without_an_error_is_not_accepted(client, internet):
    body = _data()

    def short(request):
        return httpx.Response(200, content=body[:1000], headers={"content-length": str(len(body))})

    _serve(internet, "huggingface.co", short)
    job = await _wait(
        client, (await _pull(client, spec="https://huggingface.co/o/m.gguf")).json()["id"]
    )
    assert job["status"] == "failed" and "stopped at 1000 of" in job["error"]
    assert not (client.dir / "m.gguf").exists()
    assert (client.dir / "m.gguf.part").stat().st_size == 1000  # kept: the same pull resumes


async def test_a_name_that_exists_is_refused_in_the_job_when_the_name_comes_from_the_hub(
    client, hub
):
    (client.dir / "m-q4_k_m.gguf").write_bytes(b"GGUF old")
    job = await _wait(client, (await _pull(client, spec="org/model:q4_k_m")).json()["id"])
    assert job["status"] == "failed" and "already exists" in job["error"]
    assert (client.dir / "m-q4_k_m.gguf").read_bytes() == b"GGUF old"
    forced = await _wait(
        client, (await _pull(client, spec="org/model:q4_k_m", force=True)).json()["id"]
    )
    assert forced["status"] == "done"
