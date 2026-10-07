"""A successful job poll prints no access line (the terminal filled with
`GET /jobs/<id> 200` lines while `start.sh` followed a job). Errors and every other request
keep their line. The records are built the way uvicorn's h11 and httptools protocols build them.
"""

import logging

import pytest
import uvicorn

from app import logging_setup as ls

ACCESS_FORMAT = '%s - "%s %s HTTP/%s" %d'


@pytest.fixture
def access(caplog):
    uvicorn.Config(lambda scope, receive, send: None, log_level="info")  # uvicorn's own setup
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    ls.quiet_job_polls()
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    yield logger, caplog
    logger.removeHandler(caplog.handler)
    logger.filters[:] = before


def _lines(access_fixture, method, path, status):
    logger, caplog = access_fixture
    caplog.clear()
    logger.info(ACCESS_FORMAT, "127.0.0.1:55420", method, path, "1.1", status)
    return [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize(
    "path", ["/jobs/d9f510f6cc11", "/jobs/host-1a2b", "/ui/jobs/d9f510f6cc11/status", "/jobs/x?y=1"]
)
def test_a_successful_job_poll_prints_nothing(access, path):
    assert _lines(access, "GET", path, 200) == []


def test_a_redirected_job_poll_prints_nothing(access):
    assert _lines(access, "GET", "/jobs/d9f510f6cc11", 307) == []


@pytest.mark.parametrize("status", [401, 404, 500])
def test_a_failed_job_poll_keeps_its_line(access, status):
    assert len(_lines(access, "GET", "/jobs/d9f510f6cc11", status)) == 1


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/jobs"),  # the list of jobs
        ("POST", "/jobs/d9f510f6cc11/cancel"),
        ("DELETE", "/jobs/d9f510f6cc11"),
        ("GET", "/jobs/d9f510f6cc11/log"),
        ("GET", "/ui/jobs/d9f510f6cc11"),  # the page a person opens
        ("GET", "/users"),
    ],
)
def test_every_other_request_keeps_its_line(access, method, path):
    assert len(_lines(access, method, path, 200)) == 1


def test_installing_twice_adds_one_filter(access):
    logger, _ = access
    ls.quiet_job_polls()
    assert sum(isinstance(f, ls._JobPollFilter) for f in logger.filters) == 1


def test_the_app_installs_it_after_uvicorn_is_configured():
    import inspect

    from app import main

    source = inspect.getsource(main)
    assert source.index("uvicorn.Config(") < source.index("quiet_job_polls()")
