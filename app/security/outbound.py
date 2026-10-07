"""The guard on every request the application makes on an administrator's say-so.

A request made by the server, from inside the owner's network, is a way in for whoever can
ask for it (server-side request forgery). Before anything is fetched, `Guard.prepare`:
- requires https (http only when the guard is built to allow it, which production never is);
- requires the host to be on the allow-list, itself or a parent domain of it;
- resolves the host once and requires every address to be public: no private, loopback,
  link-local, reserved or multicast address, so `127.0.0.1`, `169.254.169.254` and a name
  that points at the inside are all refused;
- returns the address to connect to, so the connection goes where the check looked (a name
  that changes its answer between the check and the connection is no way round it).
Redirects are followed by the caller one hop at a time, each hop going through `prepare` again.
"""

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class OutboundError(Exception):
    """The request was refused; the message is safe to show the administrator."""


_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_SIX_TO_FOUR = ipaddress.ip_network("2002::/16")


def is_public(address: str) -> bool:
    """True for an address on the public internet. `is_global` alone is not enough: it says
    yes to multicast, and to a NAT64 or 6to4 address that carries a private IPv4 address.
    """
    ip = ipaddress.ip_address(address)
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return is_public(str(ip.ipv4_mapped))
        if ip in _NAT64:
            return is_public(str(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)))
        if ip in _SIX_TO_FOUR:
            return is_public(str(ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF)))
    return ip.is_global and not ip.is_multicast


def system_resolver(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


@dataclass(frozen=True)
class Target:
    """Where to connect and what to say: the url with the resolved address in it, the Host
    header of the original name, and the name to verify the certificate against."""

    url: str
    host_header: str
    sni: str | None


@dataclass(frozen=True)
class Guard:
    allowed_hosts: frozenset
    allow_http: bool = False
    allow_private: bool = False
    resolver: Callable = field(default=system_resolver, compare=False)

    def host_allowed(self, host: str) -> bool:
        """The host itself or a parent domain is on the list; "*" allows any host, never
        a non-public address (checked after resolution, whatever the list says)."""
        host = host.lower()
        if "*" in self.allowed_hosts:
            return True
        return any(host == h or host.endswith("." + h) for h in self.allowed_hosts)

    def prepare(self, url: str) -> Target:
        parts = urlsplit(url)
        if parts.scheme not in ("https", "http") or not parts.hostname:
            raise OutboundError("only http(s) URLs with a host are accepted")
        if parts.scheme == "http" and not self.allow_http:
            raise OutboundError("the URL must use https")
        if parts.username or parts.password:
            raise OutboundError("a URL with credentials is refused")
        host = parts.hostname.lower()
        if not self.host_allowed(host):
            allowed = ", ".join(sorted(self.allowed_hosts))
            raise OutboundError(f"host {host} is not allowed; allowed hosts: {allowed}")
        port = parts.port or (443 if parts.scheme == "https" else 80)
        try:
            addresses = [str(ipaddress.ip_address(host))]
        except ValueError:
            try:
                addresses = self.resolver(host, port)
            except OSError:
                raise OutboundError(f"host {host} could not be resolved") from None
        if not addresses:
            raise OutboundError(f"host {host} could not be resolved")
        if not self.allow_private:
            for address in addresses:
                if not is_public(address):
                    raise OutboundError(f"host {host} points at a non-public address")
        address = addresses[0]
        literal = f"[{address}]" if ":" in address else address
        netloc = literal + (f":{parts.port}" if parts.port else "")
        pinned = parts._replace(netloc=netloc).geturl()
        host_header = parts.hostname + (f":{parts.port}" if parts.port else "")
        sni = host if parts.scheme == "https" else None
        return Target(url=pinned, host_header=host_header, sni=sni)


def guard_from_settings() -> Guard:
    """The guard a model pull uses: the hub, the extra hosts, https only, public addresses."""
    from app.config import get_settings

    settings = get_settings()
    hosts = {"huggingface.co", "hf.co"}
    hub = urlsplit(settings.model_hub_url).hostname
    if hub:
        hosts.add(hub.lower())
    extra = settings.model_pull_allowed_hosts.split(",")
    hosts.update(h.strip().lower() for h in extra if h.strip())
    return Guard(allowed_hosts=frozenset(hosts))
