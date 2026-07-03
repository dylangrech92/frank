"""Outbound HTTP for the agent: one SSRF gate in front of one TLS-verified fetch path.

All public URLs are checked against ``BLOCKED_NETS`` before any TCP connection is
opened, and TLS certificate verification is never disabled.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse
from typing import Any

import requests

# ---------------------------------------------------------------------------
# SSRF guard – networks that must never be reached by outbound fetches
# ---------------------------------------------------------------------------

_BLOCKED_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("0.0.0.0/8"),  # this host
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918
    ipaddress.ip_network("100.64.0.0/10"),  # carrier-grade NAT
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud metadata
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
)


class FetchBlocked(Exception):
    """Raised when the SSRF guard refuses a destination (private/internal host, or unresolvable)."""


def resolve_and_check(hostname: str) -> tuple[bool, str]:
    """Resolve *hostname* and verify every resolved IP is not in a blocked range.

    Args:
        hostname: The hostname to resolve.

    Returns:
        ````(True, "")`` if the hostname resolves to an allowed IP, or ````(False, reason)``
        on failure (empty hostname, DNS failure, or blocked IP).
    """
    if not hostname:
        return (False, "empty hostname")

    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return (False, f"DNS resolution failed: {hostname}")

    for info in infos:
        ip = info[4][0]
        addr = ipaddress.ip_address(ip)
        if any(addr in net for net in _BLOCKED_NETS):
            return (False, f"blocked IP range: {ip} ({hostname})")

    return (True, "")


def is_private_url(url: str) -> bool:
    """Return ``True`` when *url* resolves to a private/internal host or an invalid address.

    This function fails closed — the caller should treat the result as a rejection signal.

    Args:
        url: The URL to check.

    Returns:
        ``False`` when the URL is safe to fetch, ``True`` otherwise.
    """
    hostname = urlparse(url).hostname
    if not hostname:
        return True  # fail closed
    return not resolve_and_check(hostname)[0]


# ---------------------------------------------------------------------------
# Default headers that make us look like a regular browser
# ---------------------------------------------------------------------------

_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_page(url: str, timeout: float = 15.0) -> tuple[str, str]:
    """Fetch *url* using TLS-verified HTTPS with a new ``requests.Session``.

    TLS verification is ALWAYS on by design — there is deliberately no parameter
    to disable it. The SSRF gate runs before any connection is opened.

    Args:
        url: The public HTTP(S) URL to fetch.
        timeout: Request timeout in seconds (default 15).

    Returns:
        A ``(body, content_type)`` tuple where *content_type* is the lower-cased
        ``Content-Type`` header with any ``;charset`` suffix stripped.

    Raises:
        FetchBlocked: When the URL is private or internal.
        requests.exceptions.HTTPError: On non-2xx/3xx status codes after redirects.
    """
    if is_private_url(url):
        raise FetchBlocked(f"private or internal URL blocked: {url}")

    session = requests.Session()
    session.headers.update(_HEADERS)

    response = session.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        verify=True,
    )
    response.raise_for_status()

    content_type = ""
    if "content-type" in response.headers:
        # Lower-case and strip any charset parameter ("text/html; charset=UTF-8" -> "text/html")
        content_type = response.headers["content-type"].lower().split(";")[0].strip()

    session.close()
    return (response.text, content_type)
