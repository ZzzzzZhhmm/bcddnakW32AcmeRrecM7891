#!/usr/bin/env python3
"""Download one immutable artifact with parallel HTTP ranges and SHA-256.

This is used only for the 49.6 MB SAPIEN beta wheel that is absent from common
PyPI mirrors.  Completed range parts live beside the persistent AFS
wheelhouse, so an interrupted CCI session can resume without restarting the
whole wheel.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import shutil
import time
import urllib.request
from pathlib import Path


def chunk_ranges(size: int, workers: int) -> tuple[tuple[int, int], ...]:
    if size <= 0:
        raise ValueError("size must be positive")
    workers = max(1, min(workers, size))
    chunk = (size + workers - 1) // workers
    return tuple(
        (start, min(size - 1, start + chunk - 1))
        for start in range(0, size, chunk)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_part(
    *,
    url: str,
    path: Path,
    start: int,
    end: int,
    retries: int,
) -> None:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return
    path.unlink(missing_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "User-Agent": "WARM-runtime-bootstrap/2",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                if status != 206:
                    raise RuntimeError(
                        f"server returned HTTP {status}, expected 206"
                    )
                with temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"range {start}-{end} has {temporary.stat().st_size} "
                    f"bytes, expected {expected}"
                )
            os.replace(temporary, path)
            return
        except Exception as error:  # Network retry is intentional here.
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(8, 2**attempt))
    raise RuntimeError(f"failed range {start}-{end}: {last_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_sha256 = args.sha256.lower()
    if (
        output.is_file()
        and output.stat().st_size == args.size
        and file_sha256(output) == expected_sha256
    ):
        print(f"verified_download_cached={output}")
        return 0

    output.unlink(missing_ok=True)
    part_root = output.with_name(output.name + ".parts")
    part_root.mkdir(parents=True, exist_ok=True)
    ranges = chunk_ranges(args.size, args.workers)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(ranges)
    ) as executor:
        futures = [
            executor.submit(
                _download_part,
                url=args.url,
                path=part_root / f"part-{index:03d}",
                start=start,
                end=end,
                retries=args.retries,
            )
            for index, (start, end) in enumerate(ranges)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as destination:
        for index in range(len(ranges)):
            with (part_root / f"part-{index:03d}").open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    actual_size = temporary.stat().st_size
    actual_sha256 = file_sha256(temporary)
    if actual_size != args.size or actual_sha256 != expected_sha256:
        temporary.unlink(missing_ok=True)
        shutil.rmtree(part_root, ignore_errors=True)
        raise SystemExit(
            "download verification failed: "
            f"size={actual_size}/{args.size} "
            f"sha256={actual_sha256}/{expected_sha256}"
        )
    os.replace(temporary, output)
    shutil.rmtree(part_root)
    print(f"verified_download={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
