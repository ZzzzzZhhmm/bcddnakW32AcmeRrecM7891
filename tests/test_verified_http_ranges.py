from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_verified_http_ranges.py"
SPEC = importlib.util.spec_from_file_location("verified_ranges", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


def test_chunk_ranges_cover_file_exactly_without_overlap() -> None:
    ranges = downloader.chunk_ranges(49_596_610, 8)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 49_596_609
    assert sum(end - start + 1 for start, end in ranges) == 49_596_610
    assert all(
        previous[1] + 1 == following[0]
        for previous, following in zip(ranges, ranges[1:])
    )


def test_file_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"WARM")
    assert downloader.file_sha256(artifact) == (
        "5ed5f3cd246ce3864fc47868a167d41fe88687ca396662e8ea0581c3bb1787bc"
    )
