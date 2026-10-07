"""Tests: how the container reaches the host helper. File checks only, no Docker;
the real runs (Docker Desktop port, Linux socket with group access) are on the issue."""

from pathlib import Path

import yaml

from app.settings_rules import helper_url_problem

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())["services"]["channelagent"]
LINUX = yaml.safe_load((REPO_ROOT / "docker-compose.host-helper-linux.yml").read_text())
APP = LINUX["services"]["channelagent"]


def test_docker_desktop_reaches_the_helper_on_the_hosts_loopback():
    url = BASE["environment"]["HOST_HELPER_URL"]
    assert url == "http://host.docker.internal:${HOST_HELPER_PORT:-8701}"
    assert helper_url_problem(url.replace("${HOST_HELPER_PORT:-8701}", "8701")) is None


def test_the_helper_port_is_not_published():
    assert not any("8701" in p or "HOST_HELPER" in p for p in BASE["ports"])


def test_linux_uses_the_socket_directory_and_its_group_only():
    assert APP["environment"]["HOST_HELPER_URL"] == "unix:///run/channelagent-host/helper.sock"
    assert APP["volumes"] == ["./run:/run/channelagent-host"]
    assert APP["group_add"] == [
        "${HOST_HELPER_GID:?set HOST_HELPER_GID to the group of the socket (id -g)}"
    ]
    assert "ports" not in APP and list(LINUX["services"]) == ["channelagent"]
