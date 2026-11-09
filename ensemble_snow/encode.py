"""Compact binary bundles for the browser: one gzip file per (ensemble, method).

Layout of the (decompressed) file::

    b"ESN1" | uint32 little-endian header length | header JSON (utf-8) | block payloads

Each block is one n-dimensional array, quantised to uint8 / uint16 (``value = lo + step * code``; the top code means "missing"),
delta-coded along x and then y (modulo 2**8 / 2**16, which typed arrays in the browser reproduce for free), and, for 16-bit data,
byte-shuffled (all high bytes, then all low bytes) so that smooth fields compress well.  The browser (``site/js/bundle.js``) undoes the
three steps; :func:`read_bundle` here is the reference decoder used by the tests.
"""
from __future__ import annotations

import gzip
import json
import struct

import numpy as np

MAGIC = b"ESN1"


def quantize(a: np.ndarray, lo: float, step: float, dtype: str) -> np.ndarray:
    top = 255 if dtype == "u8" else 65535
    a = np.asarray(a, np.float64)
    with np.errstate(invalid="ignore"):
        q = np.clip(np.rint((a - lo) / step), 0, top - 1)
    q = np.where(np.isfinite(a), q, top)
    return q.astype(np.uint8 if dtype == "u8" else np.uint16)


def dequantize(q: np.ndarray, lo: float, step: float, dtype: str) -> np.ndarray:
    top = 255 if dtype == "u8" else 65535
    out = lo + step * q.astype(np.float32)
    out[q == top] = np.nan
    return out


def _delta(q: np.ndarray) -> np.ndarray:
    a = q.copy()
    a[..., :, 1:] -= q[..., :, :-1]
    b = a.copy()
    b[..., 1:, :] -= a[..., :-1, :]
    return b


def _undelta(b: np.ndarray) -> np.ndarray:
    return np.cumsum(np.cumsum(b, axis=-2, dtype=b.dtype), axis=-1, dtype=b.dtype)


def _payload(q: np.ndarray) -> bytes:
    d = _delta(q)
    if d.dtype == np.uint16:
        return np.concatenate([(d >> 8).astype(np.uint8).ravel(), (d & 0xFF).astype(np.uint8).ravel()]).tobytes()
    return d.tobytes()


def write_bundle(path, blocks: list[dict], meta: dict) -> int:
    """``blocks``: dicts with ``name``, ``data`` (float array, NaN = missing), ``dtype`` ("u8"|"u16"), ``lo``, ``step``.  Returns the file size."""
    head_blocks, payloads, offset = [], [], 0
    for b in blocks:
        q = quantize(b["data"], b["lo"], b["step"], b["dtype"])
        pl = _payload(q)
        head_blocks.append({"name": b["name"], "dtype": b["dtype"], "dims": list(q.shape), "lo": b["lo"], "step": b["step"],
                            "missing": 255 if b["dtype"] == "u8" else 65535, "offset": offset, "nbytes": len(pl)})
        payloads.append(pl)
        offset += len(pl)
    head = json.dumps({"blocks": head_blocks, "meta": meta}, separators=(",", ":")).encode()
    raw = MAGIC + struct.pack("<I", len(head)) + head + b"".join(payloads)
    blob = gzip.compress(raw, compresslevel=9, mtime=0)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob)


def read_bundle(path) -> tuple[dict, dict[str, np.ndarray]]:
    raw = gzip.decompress(open(path, "rb").read())
    if raw[:4] != MAGIC:
        raise ValueError("not an ESN1 bundle")
    (hl,) = struct.unpack("<I", raw[4:8])
    head = json.loads(raw[8:8 + hl])
    base = 8 + hl
    out = {}
    for b in head["blocks"]:
        n = int(np.prod(b["dims"]))
        buf = np.frombuffer(raw, np.uint8, b["nbytes"], base + b["offset"])
        if b["dtype"] == "u16":
            d = ((buf[:n].astype(np.uint16) << 8) | buf[n:2 * n].astype(np.uint16)).reshape(b["dims"])
        else:
            d = buf[:n].reshape(b["dims"]).copy()
        out[b["name"]] = dequantize(_undelta(d), b["lo"], b["step"], b["dtype"])
    return head["meta"], out
