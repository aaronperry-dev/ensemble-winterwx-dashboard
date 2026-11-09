"""Tiny HTTP layer: a shared session with retries, byte-range reads and HEAD probes.  Tests replace :class:`Fetcher` with an in-memory fake."""
from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)
USER_AGENT = "ensemble-snow-dashboard (github pages; open-data client)"


class NotFound(Exception):
    pass


class Fetcher:
    def __init__(self, workers: int = 16, timeout: float = 120.0, tries: int = 4):
        self.timeout, self.tries = timeout, tries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers * 2)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(self, method: str, url: str, headers: dict | None = None) -> requests.Response:
        last: Exception | None = None
        for k in range(self.tries):
            try:
                r = self.session.request(method, url, headers=headers, timeout=self.timeout)
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
