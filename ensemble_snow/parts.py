"""A resumable download: each finished forecast hour of a run is kept on disk (already cropped to the display region, a few MB at most), so a
build that is interrupted - or that loses its connection half way - only fetches the hours that are still missing when it is run again.

Layout: ``<cache>/parts/<system>/<code fingerprint>_<run>/f<hour>.pkl.gz``.  Only the newest run of a system is kept."""
from __future__ import annotations

import gzip
import logging
import pickle
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class PartCache:
    def __init__(self, root: Path, system: str, tag: str):
        self.dir = Path(root) / "parts" / system / tag
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.parent.iterdir():                  # other runs / other code versions are of no use any more
            if old != self.dir:
                shutil.rmtree(old, ignore_errors=True)

    def _path(self, step: int) -> Path:
        return self.dir / f"f{step:03d}.pkl.gz"

    def get(self, step: int):
        p = self._path(step)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
        except Exception as exc:                                # a damaged file is simply fetched again
            log.warning("part %s unreadable (%s); fetching it again", p.name, exc)
            p.unlink(missing_ok=True)
            return None

    def put(self, step: int, obj) -> None:
        p = self._path(step)
        tmp = p.with_suffix(".tmp")
        try:
            with gzip.open(tmp, "wb", compresslevel=3) as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(p)
        except Exception:
            log.exception("could not keep forecast hour %d of the download (the build goes on)", step)
            tmp.unlink(missing_ok=True)

    def count(self) -> int:
        return sum(1 for _ in self.dir.glob("f*.pkl.gz"))
