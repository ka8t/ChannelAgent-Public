"""Tests: the reference TLS reverse proxy overlay keeps the API
private by default. File checks only, no Docker: the real run (HTTPS 401/200,
LAN address refused) is recorded in the issue.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLAY = yaml.safe_load((REPO_ROOT / "docker-compose.tls.yml").read_text())
CADDYFILE = (REPO_ROOT / "docker" / "Caddyfile").read_text()
# Directives only: the file's comments mention `tls internal` too.
DIRECTIVES = [
    line.strip()
    for line in CADDYFILE.splitlines()
    if line.strip() and not line.strip().startswith("#")
]
PROXY = OVERLAY["services"]["tls-proxy"]


def test_the_proxy_is_published_on_loopback_by_default():
    assert PROXY["ports"] == ["${TLS_BIND_ADDRESS:-127.0.0.1}:${TLS_PORT:-8443}:8443"]


def test_only_the_tls_port_is_published_no_plain_http_port():
    published = [p.rsplit(":", 1)[-1] for p in PROXY["ports"]]
    assert published == ["8443"]
    assert not any(p.endswith((":80", ":443")) for p in PROXY["ports"])


def test_the_overlay_does_not_touch_the_apis_own_publishing():
    assert "channelagent" not in OVERLAY["services"]


def test_the_proxy_waits_for_the_application_and_keeps_its_certificates():
    assert PROXY["depends_on"] == ["channelagent"]
    assert "tls_proxy_data:/data" in PROXY["volumes"]
    assert "tls_proxy_data" in OVERLAY["volumes"]


def test_the_caddyfile_is_mounted_read_only():
    assert "./docker/Caddyfile:/etc/caddy/Caddyfile:ro" in PROXY["volumes"]


def test_the_caddyfile_terminates_tls_and_forwards_to_the_application():
    assert "tls internal" in DIRECTIVES
    assert "reverse_proxy channelagent:{$API_SERVER_PORT:8700}" in DIRECTIVES
    assert DIRECTIVES[0] == "{$TLS_SITE_ADDRESS:localhost}:8443 {"


def test_the_proxy_and_the_caddyfile_agree_on_the_variables():
    for name in ("TLS_SITE_ADDRESS", "API_SERVER_PORT"):
        assert name in PROXY["environment"]
        assert "{$" + name in CADDYFILE
