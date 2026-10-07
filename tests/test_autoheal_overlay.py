"""Tests: the autoheal overlay only watches the application container. File
checks, no Docker; the real run (frozen container restarted) is in the issue.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OVERLAY = yaml.safe_load((REPO_ROOT / "docker-compose.autoheal.yml").read_text())
WATCHER = OVERLAY["services"]["autoheal"]


def test_the_application_service_is_labelled_and_nothing_else_is_changed_on_it():
    assert OVERLAY["services"]["channelagent"] == {"labels": {"autoheal": "true"}}


def test_the_watcher_only_acts_on_labelled_containers():
    assert WATCHER["environment"]["AUTOHEAL_CONTAINER_LABEL"] == "autoheal"


def test_the_watcher_gets_the_docker_socket_and_nothing_else_from_the_host():
    assert WATCHER["volumes"] == ["/var/run/docker.sock:/var/run/docker.sock"]
    assert "ports" not in WATCHER and "network_mode" not in WATCHER


def test_the_watcher_restarts_itself_and_checks_often_enough():
    assert WATCHER["restart"] == "unless-stopped"
    assert int(WATCHER["environment"]["AUTOHEAL_INTERVAL"]) <= 30


def test_the_base_compose_file_is_unchanged_by_the_overlay():
    base = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    assert "labels" not in base["services"]["channelagent"]
    assert "autoheal" not in base["services"]
