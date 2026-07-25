#!/usr/bin/env python3
"""Run the f77c633 RMBench contract builder with one bounded local repair.

The attested training commit contains a contract-only bug: ``_build_contract``
validates the encoder contract but drops the returned compute device before
using that name in the RMBench runtime-projection branch.  This launcher keeps
the historical source tree immutable, verifies the exact two affected source
blobs, loads that historical bundle, and injects only the missing module global
that Python would otherwise fail to resolve.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType
from typing import Sequence


EXPECTED_COMMIT = "f77c63385c747fdc1424386489cbf9c7ea57ddc5"
EXPECTED_ONLINE_BUILDER_SHA256 = (
    "f205796a2ef2e9ad58f7a3e35e95473926ea51bf8833d2b583858414842eb4a5"
)
EXPECTED_BUNDLE_BUILDER_SHA256 = (
    "a516d3ab2a8d3bbef2654a6d5a972d86085e2754112f54bf7e616e3ac133c109"
)


class CompatibilityError(RuntimeError):
    """Raised before running code outside the exact admitted source pair."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_historical_bundle(root: Path) -> ModuleType:
    online_path = root / "scripts" / "build_warm_online_contract.py"
    bundle_path = root / "scripts" / "build_warm_rmbench_contract_bundle.py"
    if _git_head(root) != EXPECTED_COMMIT:
        raise CompatibilityError("historical evaluation checkout commit mismatch")
    expected = (
        (online_path, EXPECTED_ONLINE_BUILDER_SHA256),
        (bundle_path, EXPECTED_BUNDLE_BUILDER_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or _sha256(path) != digest:
            raise CompatibilityError(
                f"historical contract source hash mismatch: {path}"
            )

    root_text = str(root)
    src_text = str(root / "src")
    for path in (root_text, src_text):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, src_text)
    sys.path.insert(0, root_text)

    spec = importlib.util.spec_from_file_location(
        "_warm_f77_rmbench_contract_bundle",
        bundle_path,
    )
    if spec is None or spec.loader is None:
        raise CompatibilityError("cannot load historical RMBench bundle builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    online_builder = getattr(module, "online_builder", None)
    loaded_path = Path(getattr(online_builder, "__file__", "")).resolve()
    if loaded_path != online_path.resolve():
        raise CompatibilityError(
            f"bundle imported the wrong online builder: {loaded_path}"
        )
    return module


def _encoder_contract_from_argv(argv: Sequence[str]) -> Path:
    matches = [
        Path(argv[index + 1]).expanduser().resolve()
        for index, value in enumerate(argv[:-1])
        if value == "--encoder-contract"
    ]
    if len(matches) != 1:
        raise CompatibilityError(
            "contract invocation must contain exactly one --encoder-contract"
        )
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    historical_root_raw = os.environ.get("WARM_EVAL_COMPAT_HISTORICAL_ROOT", "")
    if not historical_root_raw:
        raise CompatibilityError("WARM_EVAL_COMPAT_HISTORICAL_ROOT is required")
    historical_root = Path(historical_root_raw).expanduser().resolve()
    module = _load_historical_bundle(historical_root)
    online_builder = module.online_builder

    encoder_path = _encoder_contract_from_argv(arguments)
    encoder_contract = json.loads(encoder_path.read_text(encoding="utf-8"))
    if not isinstance(encoder_contract, dict):
        raise CompatibilityError("encoder contract must contain a JSON object")
    _, _, _, compute_device = online_builder.validate_online_encoder_contract(
        encoder_contract
    )
    if re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", str(compute_device)) is None:
        raise CompatibilityError(
            f"encoder contract returned an unsafe compute device: {compute_device!r}"
        )

    # This is the complete compatibility repair. The historical function has
    # no local assignment named compute_device, so its unresolved global lookup
    # now receives the validated value that the original call accidentally
    # discarded.
    online_builder.compute_device = compute_device
    print(
        "rmbench_contract_compat_ok "
        f"commit={EXPECTED_COMMIT} compute_device={compute_device}"
    )
    return int(module.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
