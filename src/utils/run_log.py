from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, TextIO


class _Tee(TextIO):
    def __init__(self, primary: TextIO, *extra: TextIO) -> None:
        self._primary = primary
        self._extra = extra

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        for s in self._extra:
            try:
                s.write(data)
                s.flush()
            except (ValueError, OSError):
                pass
        self._primary.flush()
        return n

    def flush(self) -> None:
        self._primary.flush()
        for s in self._extra:
            try:
                s.flush()
            except (ValueError, OSError):
                pass

    def isatty(self) -> bool:
        return self._primary.isatty()


@contextmanager
def tee_stdout_txt(
    log_path: str | Path,
    *,
    append: bool = False,
    encoding: str = "utf-8",
) -> Generator[None, None, None]:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding=encoding) as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
            except (ValueError, OSError):
                pass
            sys.stdout = old
