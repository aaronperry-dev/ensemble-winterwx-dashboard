"""Ensemble snowfall dashboard pipeline."""


def _silence_progress_bars() -> None:
    """Switch every tqdm progress bar off.

    ``ecmwf-opendata`` (through ``multiurl``) draws one bar per download, and in a CI log each redraw is a separate line: a full EPS run came to ~57 000
    lines / 4.6 MB, which made the live log unusable and the job's final bookkeeping heavy.  The environment variable ``TQDM_DISABLE`` is not honoured by
    every tqdm version, so the class itself is patched.  Set ``ENSNOW_PROGRESS=1`` to see the bars when running locally.
    """
    import os
    if os.environ.get("ENSNOW_PROGRESS", "").lower() in ("1", "true", "yes"):
        return
    try:
        from tqdm import std
    except Exception:            # tqdm not installed: nothing to silence
        return
    if getattr(std.tqdm.__init__, "_ensnow_silent", False):
        return
    original = std.tqdm.__init__

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        original(self, *args, **kwargs)

    __init__._ensnow_silent = True
    std.tqdm.__init__ = __init__


_silence_progress_bars()
