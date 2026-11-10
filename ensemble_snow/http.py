"""Tiny HTTP layer: a shared session with retries, byte-range reads and HEAD probes.  Tests replace :class:`Fetcher` with an in-memory fake.

A server that cannot even be *connected to* (a blocked or throttled address, an outage) must not hold a build up: connections give up after
``connect_timeout`` seconds, and after ``trip_after`` connection failures in a row without a single success the host is treated as down for
``cooldown`` seconds, during which every request to it fails at once with :class:`HostDown` so that a caller can switch to a mirror."""
from __future__ import annotations

import logging
import threading
import time
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)
USER_AGENT = "ensemble-snow-dashboard (github pages; open-data client)"


class NotFound(Exception):
    pass


class HostDown(RuntimeError):
    """The host has refused every connection for a while; no request is sent until the cool-down is over."""


class Fetcher:
    def __init__(self, workers: int = 16, timeout: float = 120.0, tries: int = 4, connect_timeout: float = 15.0,
                 trip_after: int = 6, cooldown: float = 60.0):
        self.timeout, self.tries = (connect_timeout, timeout), tries
        self.trip_after, self.cooldown = trip_after, cooldown
        self._fails: dict[str, int] = {}
        self._down_until: dict[str, float] = {}
        self._lock = threading.Lock()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers * 2)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # -- circuit breaker ------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _host(url: str) -> str:
        return urlsplit(url).netloc

    def host_down(self, url: str) -> bool:
        with self._lock:
            return time.monotonic() < self._down_until.get(self._host(url), 0.0)

    def _note_ok(self, host: str) -> None:
        with self._lock:
            self._fails[host] = 0
            self._down_until.pop(host, None)

    def _note_connect_failure(self, host: str) -> bool:
        """Count a failed connection; True when the host has just been (or already is) declared down."""
        with self._lock:
            n = self._fails.get(host, 0) + 1
            self._fails[host] = n
            if n >= self.trip_after:
                if time.monotonic() >= self._down_until.get(host, 0.0):
                    log.warning("%s: %d connection attempts in a row failed; not trying it again for %.0f s", host, n, self.cooldown)
                self._down_until[host] = time.monotonic() + self.cooldown
                self._fails[host] = self.trip_after - 1             # after the cool-down a single failure trips it again
                return True
            return False

    # -- requests -------------------------------------------------------------------------------------------------------------------
    def _request(self, method: str, url: str, headers: dict | None = None) -> requests.Response:
        host = self._host(url)
        last: Exception | None = None
        for k in range(self.tries):
            if self.host_down(url):
                raise HostDown(f"{host} is not answering: {last or 'earlier connection failures'}")
            try:
                r = self.session.request(method, url, headers=headers, timeout=self.timeout)
                self._note_ok(host)                                   # the server answered, whatever it said
                if r.status_code == 404:
                    raise NotFound(url)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} for {url}")
                r.raise_for_status()
                return r
            except NotFound:
                raise
            except Exception as exc:                                  # network hiccup: back off and try again
                last = exc
                if isinstance(exc, requests.ConnectionError) and self._note_connect_failure(host):
                    raise HostDown(f"{host} is not answering: {exc}") from None
                time.sleep(min(2 ** k, 20))
        raise RuntimeError(f"giving up on {url}: {last}")

    def get(self, url: str, byte_range: tuple[int, int | None] | None = None) -> bytes:
        headers = None
        if byte_range is not None:
            a, b = byte_range
            headers = {"Range": f"bytes={a}-{'' if b is None else b}"}
        return self._request("GET", url, headers).content

    def exists(self, url: str) -> bool:
        try:
            self._request("HEAD", url)
            return True
        except NotFound:
            return False
        except Exception:
            return False
