"""Tests: the guard on requests made on an administrator's say-so."""

import pytest

from app.security.outbound import Guard, OutboundError, guard_from_settings

PUBLIC = "93.184.216.34"


def make(resolved=None, hosts=("huggingface.co", "hf.co"), **kw):
    calls = []

    def resolver(host, port):
        calls.append((host, port))
        return list(resolved if resolved is not None else [PUBLIC])

    return Guard(allowed_hosts=frozenset(hosts), resolver=resolver, **kw), calls


@pytest.mark.parametrize(
    "url, fragment",
    [
        ("http://huggingface.co/x", "https"),
        ("https://127.0.0.1/x", "not allowed"),
        ("https://169.254.169.254/latest/meta-data", "not allowed"),
        ("https://[::1]/x", "not allowed"),
        ("https://evil.example/x", "allowed hosts: hf.co, huggingface.co"),
        ("https://user:pw@huggingface.co/x", "credentials"),
        ("ftp://huggingface.co/x", r"http\(s\)"),
        ("https:///x", r"http\(s\)"),
    ],
)
def test_a_bad_url_is_refused_before_anything_is_resolved(url, fragment):
    guard, calls = make()
    with pytest.raises(OutboundError, match=fragment):
        guard.prepare(url)
    assert calls == []


@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.5",
        "192.168.1.10",
        "172.16.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "224.0.0.1",
        "ff02::1",
        "64:ff9b::a00:1",
        "2002:0a00:0001::1",
        "::ffff:10.0.0.1",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
    ],
)
def test_an_allowed_name_that_points_inside_is_refused(address):
    guard, _ = make(resolved=[address])
    with pytest.raises(OutboundError, match="non-public"):
        guard.prepare("https://huggingface.co/x")


def test_one_private_address_among_public_ones_refuses_the_name():
    guard, _ = make(resolved=[PUBLIC, "10.0.0.5"])
    with pytest.raises(OutboundError, match="non-public"):
        guard.prepare("https://huggingface.co/x")


def test_a_name_that_does_not_resolve_is_refused():
    def broken(host, port):
        raise OSError("no such host")

    guard = Guard(allowed_hosts=frozenset({"huggingface.co"}), resolver=broken)
    with pytest.raises(OutboundError, match="could not be resolved"):
        guard.prepare("https://huggingface.co/x")


def test_a_good_url_is_pinned_to_the_address_that_was_checked():
    guard, calls = make()
    target = guard.prepare("https://huggingface.co/Qwen/m/resolve/main/f.gguf?download=true")
    assert calls == [("huggingface.co", 443)]  # resolved exactly once
    assert target.url == f"https://{PUBLIC}/Qwen/m/resolve/main/f.gguf?download=true"
    assert target.host_header == "huggingface.co"
    assert target.sni == "huggingface.co"


def test_a_port_is_kept_and_an_ipv6_address_is_bracketed():
    guard, calls = make(resolved=["2606:4700::6810:84e5"])
    target = guard.prepare("https://huggingface.co:8443/x")
    assert calls == [("huggingface.co", 8443)]
    assert target.url == "https://[2606:4700::6810:84e5]:8443/x"
    assert target.host_header == "huggingface.co:8443"


def test_a_subdomain_of_an_allowed_host_is_allowed_but_a_lookalike_is_not():
    guard, _ = make()
    assert guard.prepare("https://cdn-lfs.huggingface.co/x").sni == "cdn-lfs.huggingface.co"
    assert guard.prepare("https://cas-bridge.xethub.hf.co/x").sni == "cas-bridge.xethub.hf.co"
    for lookalike in ("https://evilhuggingface.co/x", "https://huggingface.co.evil.example/x"):
        with pytest.raises(OutboundError, match="not allowed"):
            guard.prepare(lookalike)


def test_adding_a_host_to_the_allow_list_accepts_the_same_url():
    url = "https://models.example.net/f.gguf"
    guard, _ = make()
    with pytest.raises(OutboundError, match="allowed hosts"):
        guard.prepare(url)
    extended, _ = make(hosts=("huggingface.co", "hf.co", "models.example.net"))
    assert extended.prepare(url).sni == "models.example.net"


def test_http_and_private_addresses_are_allowed_only_when_the_guard_is_built_that_way():
    guard, _ = make(
        resolved=["127.0.0.1"], hosts=("127.0.0.1",), allow_http=True, allow_private=True
    )
    assert guard.prepare("http://127.0.0.1:8123/x").sni is None


def test_the_production_guard_allows_the_hub_and_what_the_administrator_adds(monkeypatch):
    from app.config import get_settings

    monkeypatch.delenv("MODEL_PULL_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("MODEL_HUB_URL", raising=False)
    get_settings.cache_clear()
    guard = guard_from_settings()
    assert guard.allowed_hosts == frozenset({"huggingface.co", "hf.co"})
    assert not guard.allow_http and not guard.allow_private
    monkeypatch.setenv("MODEL_PULL_ALLOWED_HOSTS", "Cdn.Example.org, models.example.net ,")
    monkeypatch.setenv("MODEL_HUB_URL", "https://hub.internal.example")
    get_settings.cache_clear()
    assert guard_from_settings().allowed_hosts == frozenset(
        {"huggingface.co", "hf.co", "cdn.example.org", "models.example.net", "hub.internal.example"}
    )
    get_settings.cache_clear()


def test_the_limits_have_defaults_of_64_gib_and_6_hours():
    from app.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    assert s.model_pull_max_bytes == 64 * 1024**3
    assert s.model_pull_timeout_seconds == 6 * 3600


@pytest.mark.parametrize(
    "address, public",
    [
        (PUBLIC, True),
        ("2606:4700::6810:84e5", True),
        ("2002:5db8:d822::1", True),  # 6to4 carrying 93.184.216.34
        ("64:ff9b::5db8:d822", True),  # NAT64 carrying 93.184.216.34
        ("::ffff:93.184.216.34", True),  # IPv4-mapped
        ("2002:0a00:0001::1", False),  # 6to4 carrying 10.0.0.1
        ("64:ff9b::a00:1", False),  # NAT64 carrying 10.0.0.1
        ("::ffff:10.0.0.1", False),
        ("224.0.0.1", False),
    ],
)
def test_an_address_that_carries_another_one_is_judged_by_the_inner_address(address, public):
    from app.security.outbound import is_public

    assert is_public(address) is public
