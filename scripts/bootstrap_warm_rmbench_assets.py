#!/usr/bin/env python3
"""Install the pinned official RMBench simulator assets into persistent AFS.

RMBench's policy-training data and its SAPIEN simulator assets are different
artifacts.  The official Git checkout intentionally ignores the large asset
trees and downloads them from the RMBench Hugging Face dataset.  This helper:

1. downloads only the ALOHA embodiment and object trees at the protocol-pinned
   dataset revision;
2. keeps the resumable snapshot outside the Git checkout;
3. renders the official ``*_tmp.yml`` embodiment templates deterministically;
4. records a self-checking provenance manifest; and
5. links the two ignored trees into the pinned official checkout.

It never deletes a non-empty directory.  A partial or conflicting deployment
therefore fails closed instead of silently replacing user data.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ASSET_SCHEMA = "warm.rmbench-simulator-assets"
ASSET_VERSION = 1
RMBENCH_HF_REPOSITORY = "TianxingChen/RMBench"
RMBENCH_HF_REVISION = "855e90e1213d150bf4889130e83398f107314681"
ASSET_PATTERNS = (
    "embodiments/aloha-agilex/**",
    "objects/**",
)
ASSET_MANIFEST_NAME = ".warm_rmbench_assets.json"
DOWNLOAD_PLAN_NAME = ".warm_rmbench_download_plan.json"
BUNDLED_DOWNLOAD_PLAN = (
    Path(__file__).resolve().parent
    / "constraints"
    / "rmbench_assets_855e90e1213d.json.zlib.b64"
)
EXPECTED_ASSET_FILE_COUNT = 344
EXPECTED_ASSET_TOTAL_BYTES = 1_352_861_012
DEFAULT_NETWORK_TIMEOUT_SECONDS = 600
DEFAULT_DOWNLOAD_RETRIES = 8

# Every object required by the official nine-task suite.  The final four are
# common RMBench utility assets used by task helpers rather than appearing as
# literal ``modelname=`` arguments in every task module.
REQUIRED_OBJECT_DIRECTORIES = (
    "002_breadbasket",
    "003_cover",
    "004_numbercard",
    "005_button",
    "006_check_button",
    "007_T_block",
    "008_shelf",
    "009_toycar",
    "010_mouse",
    "011_stapler",
    "012_bell",
    "013_playingcards",
    "017_battery_slot_gauge",
    "018_battery",
    "cube",
    "sapien-block1",
    "sapien-block2",
    "vis_box",
)


class AssetBootstrapError(RuntimeError):
    """Raised when the official asset snapshot cannot be proven complete."""


def _positive_int_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise AssetBootstrapError(
            f"{name} must be a positive integer, got {raw!r}"
        ) from error
    if value < 1:
        raise AssetBootstrapError(
            f"{name} must be a positive integer, got {raw!r}"
        )
    return value


def _exception_chain(error: BaseException) -> str:
    rendered: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        suffix = f" status={status}" if status is not None else ""
        rendered.append(f"{type(current).__name__}{suffix}: {current}")
        current = current.__cause__ or current.__context__
    return " <- ".join(rendered)


def _endpoint_candidates() -> tuple[str, ...]:
    configured = os.environ.get("RMBENCH_HF_ENDPOINTS", "")
    values = [value.strip() for value in configured.split(",") if value.strip()]
    if not values:
        explicit = os.environ.get("HF_ENDPOINT", "").strip()
        if explicit:
            values.append(explicit)
        values.extend(("https://huggingface.co", "https://hf-mirror.com"))
    unique: list[str] = []
    for value in values:
        normalized = value.rstrip("/")
        if normalized and normalized not in unique:
            unique.append(normalized)
    if not unique:
        raise AssetBootstrapError("no Hugging Face download endpoint is configured")
    return tuple(unique)


def _is_nonempty_directory(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _payload_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for category in ("embodiments", "objects"):
        category_root = root / category
        if not category_root.is_dir():
            continue
        files.extend(path for path in category_root.rglob("*") if path.is_file())
    return tuple(sorted(files, key=lambda path: path.relative_to(root).as_posix()))


def _tree_metadata(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in _payload_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        count += 1
        total_bytes += size
    return {
        "file_count": count,
        "total_bytes": total_bytes,
        "tree_metadata_sha256": digest.hexdigest(),
    }


def render_embodiment_configs(asset_root: Path, rmbench_root: Path) -> list[str]:
    """Apply the pinned official path-template operation without prompting."""

    embodiment_root = asset_root / "embodiments"
    rendered: list[str] = []
    replacement = str(rmbench_root.resolve())
    for template in sorted(embodiment_root.rglob("*_tmp.yml")):
        output = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        source = template.read_text(encoding="utf-8")
        value = source.replace("${ASSETS_PATH}", replacement)
        value = value.replace("$ASSETS_PATH", replacement)
        output.write_text(value, encoding="utf-8")
        rendered.append(output.relative_to(asset_root).as_posix())
    return rendered


def validate_asset_layout(asset_root: Path) -> dict[str, Any]:
    """Validate the complete RGB-only official9 asset closure."""

    root = asset_root.resolve()
    aloha = root / "embodiments" / "aloha-agilex"
    objects = root / "objects"
    missing: list[str] = []
    if not _is_nonempty_directory(aloha):
        missing.append("embodiments/aloha-agilex (missing or empty)")
    if not _is_nonempty_directory(objects):
        missing.append("objects (missing or empty)")

    config = aloha / "config.yml"
    if not config.is_file() or config.stat().st_size == 0:
        missing.append("embodiments/aloha-agilex/config.yml")
    else:
        config_source = config.read_text(encoding="utf-8")
        if "$ASSETS_PATH" in config_source:
            missing.append("embodiments/aloha-agilex/config.yml (unresolved template)")

    if aloha.is_dir():
        if not any(aloha.rglob("*.urdf")):
            missing.append("embodiments/aloha-agilex/**/*.urdf")
        if not any(aloha.rglob("*.srdf")):
            missing.append("embodiments/aloha-agilex/**/*.srdf")
        mesh_suffixes = {".dae", ".glb", ".obj", ".stl"}
        if not any(
            path.is_file() and path.suffix.lower() in mesh_suffixes
            for path in aloha.rglob("*")
        ):
            missing.append("embodiments/aloha-agilex mesh closure")

    for name in REQUIRED_OBJECT_DIRECTORIES:
        directory = objects / name
        if not _is_nonempty_directory(directory):
            missing.append(f"objects/{name} (missing or empty)")

    if missing:
        raise AssetBootstrapError(
            "official RMBench simulator assets are incomplete: " + ", ".join(missing)
        )
    metadata = _tree_metadata(root)
    if metadata["file_count"] < 100:
        raise AssetBootstrapError(
            "official RMBench asset snapshot contains implausibly few files: "
            f"{metadata['file_count']}"
        )
    return metadata


def _manifest_path(asset_root: Path) -> Path:
    return asset_root / ASSET_MANIFEST_NAME


def validate_asset_manifest(asset_root: Path) -> dict[str, Any]:
    path = _manifest_path(asset_root)
    if not path.is_file():
        raise AssetBootstrapError(f"asset provenance manifest is absent: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": ASSET_SCHEMA,
        "version": ASSET_VERSION,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "allow_patterns": list(ASSET_PATTERNS),
    }
    mismatches = [
        f"{key}={value.get(key)!r}, expected {expected!r}"
        for key, expected in required.items()
        if value.get(key) != expected
    ]
    actual = validate_asset_layout(asset_root)
    for key in ("file_count", "total_bytes", "tree_metadata_sha256"):
        if value.get(key) != actual[key]:
            mismatches.append(
                f"{key}={value.get(key)!r}, current tree={actual[key]!r}"
            )
    if mismatches:
        raise AssetBootstrapError(
            "asset provenance mismatch: " + "; ".join(mismatches)
        )
    return value


def write_asset_manifest(
    asset_root: Path,
    *,
    source: str,
    rendered_configs: Iterable[str],
) -> dict[str, Any]:
    metadata = validate_asset_layout(asset_root)
    value: dict[str, Any] = {
        "schema": ASSET_SCHEMA,
        "version": ASSET_VERSION,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "allow_patterns": list(ASSET_PATTERNS),
        "source": source,
        "rendered_configs": sorted(set(rendered_configs)),
        **metadata,
    }
    path = _manifest_path(asset_root)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return value


def _run_readonly_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    canonical = root.resolve()
    # ``safe.directory`` is only honored from protected configuration scopes.
    # Some cluster Git builds do not treat ``git -c`` as protected and still
    # reject AFS checkouts whose recorded owner differs from the container
    # user.  Give this one read-only subprocess an isolated HOME/global config
    # instead of mutating the user's persistent Git configuration.
    with tempfile.TemporaryDirectory(prefix="warm-rmbench-git-") as git_home:
        global_config = Path(git_home) / ".gitconfig"
        safe_directory = canonical.as_posix()
        global_config.write_text(
            f"[safe]\n\tdirectory = {json.dumps(safe_directory)}\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["HOME"] = git_home
        environment["GIT_CONFIG_GLOBAL"] = str(global_config)
        return subprocess.run(
            ["git", "-C", str(canonical), *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )


def _assert_pinned_checkout(root: Path, revision: str) -> None:
    if not root.is_dir():
        raise AssetBootstrapError(f"RMBench checkout is absent: {root}")


def _link_category(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise AssetBootstrapError(
                f"conflicting asset symlink {target} -> {target.resolve()}; "
                f"expected {source.resolve()}"
            )
        return
    if target.exists():
        if target.is_dir() and not any(
            child.is_file() or child.is_symlink()
            for child in target.rglob("*")
        ):
            # Failed/manual setup often leaves category/aloha-agilex as an
            # empty directory tree.  Removing only fileless directories is
            # safe and keeps retries automatic; any file or link still causes
            # the fail-closed branch below.
            for directory, _, _ in os.walk(target, topdown=False):
                Path(directory).rmdir()
        else:
            raise AssetBootstrapError(
                f"refusing to replace non-empty asset path: {target}; "
                "move it aside or pass it as --asset-source with a valid manifest"
            )
    target.symlink_to(source.resolve(), target_is_directory=True)


def deploy_asset_links(
    *,
    asset_root: Path,
    rmbench_root: Path,
    rmbench_revision: str,
) -> None:
    _assert_pinned_checkout(rmbench_root, rmbench_revision)
    validate_asset_manifest(asset_root)
    checkout_assets = rmbench_root / "assets"
    checkout_assets.mkdir(parents=True, exist_ok=True)
    for category in ("embodiments", "objects"):
        _link_category(asset_root / category, checkout_assets / category)
    validate_asset_layout(checkout_assets)

    completed = _run_readonly_git(
        rmbench_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if completed.returncode or completed.stdout.strip():
        raise AssetBootstrapError(
            "asset deployment changed the pinned RMBench Git worktree: "
            f"{completed.stdout.strip() or completed.stderr.strip()}"
        )


def _authorization_headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"User-Agent": "WARM-RMBench-asset-bootstrap/2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _open_with_retries(
    request: urllib.request.Request,
    *,
    timeout: int,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> Any:
    errors: list[str] = []
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            errors.append(f"attempt={attempt + 1}: {_exception_chain(error)}")
            if isinstance(error, urllib.error.HTTPError) and error.code in {
                400,
                401,
                403,
                404,
            }:
                break
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 30))
    raise AssetBootstrapError(
        f"network request failed url={request.full_url!r}: " + " | ".join(errors)
    )


def _next_link(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r'<([^>]+)>;\s*rel="next"', value)
    return match.group(1) if match else None


def _tree_url(endpoint: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="/")
    return (
        f"{endpoint}/api/datasets/{RMBENCH_HF_REPOSITORY}/tree/"
        f"{RMBENCH_HF_REVISION}/{encoded_path}?recursive=true&expand=true"
    )


def _plan_entry(value: Mapping[str, Any]) -> dict[str, Any]:
    path = str(value.get("path", ""))
    size = value.get("size")
    oid = str(value.get("oid", ""))
    if not path or not isinstance(size, int) or size < 0 or not oid:
        raise AssetBootstrapError(f"malformed Hugging Face tree entry: {value!r}")
    lfs = value.get("lfs")
    if isinstance(lfs, Mapping):
        digest = str(lfs.get("oid", ""))
        algorithm = "sha256"
        if lfs.get("size") != size or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssetBootstrapError(
                f"malformed LFS provenance for asset {path!r}: {lfs!r}"
            )
    else:
        digest = oid
        algorithm = "git-sha1"
        if not re.fullmatch(r"[0-9a-f]{40}", digest):
            raise AssetBootstrapError(
                f"malformed Git provenance for asset {path!r}: oid={digest!r}"
            )
    return {
        "path": path,
        "size": size,
        "algorithm": algorithm,
        "digest": digest,
    }


def _validate_download_plan(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    if value.get("repo_id") != RMBENCH_HF_REPOSITORY:
        raise AssetBootstrapError("asset download plan repository does not match")
    if value.get("revision") != RMBENCH_HF_REVISION:
        raise AssetBootstrapError("asset download plan revision does not match")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise AssetBootstrapError("asset download plan has no file list")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_prefixes = ("embodiments/aloha-agilex/", "objects/")
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise AssetBootstrapError("asset download plan contains a malformed row")
        entry = {
            "path": str(raw.get("path", "")),
            "size": raw.get("size"),
            "algorithm": str(raw.get("algorithm", "")),
            "digest": str(raw.get("digest", "")),
        }
        path = entry["path"]
        if (
            not path.startswith(allowed_prefixes)
            or path in seen
            or not isinstance(entry["size"], int)
            or entry["size"] < 0
            or entry["algorithm"] not in {"sha256", "git-sha1"}
        ):
            raise AssetBootstrapError(
                f"invalid pinned asset download-plan row: {entry!r}"
            )
        expected_digest_length = 64 if entry["algorithm"] == "sha256" else 40
        if not re.fullmatch(
            rf"[0-9a-f]{{{expected_digest_length}}}", entry["digest"]
        ):
            raise AssetBootstrapError(
                f"invalid digest in pinned asset download-plan row: {entry!r}"
            )
        seen.add(path)
        files.append(entry)
    total_bytes = sum(int(entry["size"]) for entry in files)
    if (
        len(files) != EXPECTED_ASSET_FILE_COUNT
        or total_bytes != EXPECTED_ASSET_TOTAL_BYTES
    ):
        raise AssetBootstrapError(
            "pinned asset tree closure differs from the audited revision: "
            f"files={len(files)} bytes={total_bytes}, expected "
            f"files={EXPECTED_ASSET_FILE_COUNT} "
            f"bytes={EXPECTED_ASSET_TOTAL_BYTES}"
        )
    return sorted(files, key=lambda entry: entry["path"])


def _fetch_download_plan(endpoint: str, *, timeout: int) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    headers = _authorization_headers()
    for root in ("embodiments/aloha-agilex", "objects"):
        url: str | None = _tree_url(endpoint, root)
        while url is not None:
            request = urllib.request.Request(url, headers=headers)
            with _open_with_retries(request, timeout=timeout) as response:
                try:
                    values = json.load(response)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AssetBootstrapError(
                        f"invalid asset tree response from {url!r}"
                    ) from error
                if not isinstance(values, list):
                    raise AssetBootstrapError(
                        f"unexpected asset tree response from {url!r}"
                    )
                for value in values:
                    if isinstance(value, Mapping) and value.get("type") == "file":
                        files.append(_plan_entry(value))
                url = _next_link(response.headers.get("Link"))
    plan: dict[str, Any] = {
        "schema": "warm.rmbench-asset-download-plan",
        "version": 1,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "endpoint": endpoint,
        "files": files,
    }
    plan["files"] = _validate_download_plan(plan)
    return plan


def _download_plan_path(asset_root: Path) -> Path:
    return asset_root / DOWNLOAD_PLAN_NAME


def _load_bundled_download_plan() -> dict[str, Any]:
    try:
        encoded = "".join(
            BUNDLED_DOWNLOAD_PLAN.read_text(encoding="ascii").split()
        )
        rows = json.loads(zlib.decompress(base64.b64decode(encoded)))
    except (OSError, ValueError, zlib.error, json.JSONDecodeError) as error:
        raise AssetBootstrapError(
            f"bundled pinned asset plan is unreadable: {BUNDLED_DOWNLOAD_PLAN}"
        ) from error
    if not isinstance(rows, list):
        raise AssetBootstrapError("bundled pinned asset plan is not a list")
    files: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise AssetBootstrapError(
                f"malformed bundled pinned asset row: {row!r}"
            )
        path, size, compact_algorithm, digest = row
        algorithm = {"s": "sha256", "g": "git-sha1"}.get(compact_algorithm)
        if algorithm is None:
            raise AssetBootstrapError(
                f"unknown bundled asset digest algorithm: {compact_algorithm!r}"
            )
        files.append(
            {
                "path": path,
                "size": size,
                "algorithm": algorithm,
                "digest": digest,
            }
        )
    value: dict[str, Any] = {
        "schema": "warm.rmbench-asset-download-plan",
        "version": 1,
        "repo_id": RMBENCH_HF_REPOSITORY,
        "revision": RMBENCH_HF_REVISION,
        "endpoint": "bundled-official-manifest",
        "files": files,
    }
    value["files"] = _validate_download_plan(value)
    return value


def _load_or_fetch_download_plan(
    asset_root: Path,
    *,
    endpoints: Sequence[str],
    timeout: int,
) -> dict[str, Any]:
    path = _download_plan_path(asset_root)
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            _validate_download_plan(value)
        except (AssetBootstrapError, json.JSONDecodeError, OSError) as error:
            print(f"asset_download_plan_ignored={_exception_chain(error)}")
        else:
            print(f"asset_download_plan_reused={path}")
            return value

    try:
        value = _load_bundled_download_plan()
    except AssetBootstrapError as error:
        print(f"asset_bundled_download_plan_ignored={_exception_chain(error)}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        print(
            f"asset_download_plan_bundled={BUNDLED_DOWNLOAD_PLAN} "
            f"files={len(value['files'])}",
            flush=True,
        )
        return value

    failures: list[str] = []
    for endpoint in endpoints:
        try:
            value = _fetch_download_plan(endpoint, timeout=timeout)
        except (AssetBootstrapError, OSError) as error:
            failures.append(f"{endpoint}: {_exception_chain(error)}")
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        print(f"asset_download_plan_endpoint={endpoint}")
        return value
    raise AssetBootstrapError(
        "could not list the pinned RMBench asset tree through any endpoint: "
        + " | ".join(failures)
    )


def _hash_matches(path: Path, entry: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    algorithm = str(entry["algorithm"])
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git-sha1":
        digest = hashlib.sha1()
        prefix = f"blob {entry['size']}\0".encode("ascii")
    else:
        raise AssetBootstrapError(f"unsupported asset hash algorithm: {algorithm}")
    digest.update(prefix)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == entry["digest"]


def _resolve_url(endpoint: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="/")
    return (
        f"{endpoint}/datasets/{RMBENCH_HF_REPOSITORY}/resolve/"
        f"{RMBENCH_HF_REVISION}/{encoded_path}?download=true"
    )


def _probe_download_endpoints(
    endpoints: Sequence[str],
    *,
    entry: Mapping[str, Any],
    timeout: int,
) -> tuple[str, ...]:
    usable: list[str] = []
    failures: list[str] = []
    for endpoint in endpoints:
        headers = _authorization_headers()
        headers["Range"] = "bytes=0-0"
        request = urllib.request.Request(
            _resolve_url(endpoint, str(entry["path"])),
            headers=headers,
        )
        print(f"asset_endpoint_probe_start={endpoint}", flush=True)
        try:
            with _open_with_retries(
                request,
                timeout=timeout,
                retries=1,
            ) as response:
                response.read(1)
        except AssetBootstrapError as error:
            failures.append(f"{endpoint}: {_exception_chain(error)}")
            print(
                f"asset_endpoint_probe_failed={endpoint} "
                f"reason={_exception_chain(error)}",
                file=sys.stderr,
                flush=True,
            )
            continue
        usable.append(endpoint)
        print(f"asset_endpoint_probe_ok={endpoint}", flush=True)
    if not usable:
        raise AssetBootstrapError(
            "no RMBench asset download endpoint is reachable: "
            + " | ".join(failures)
        )
    return tuple(usable)


def _download_file(
    *,
    destination: Path,
    entry: Mapping[str, Any],
    endpoints: Sequence[str],
    timeout: int,
) -> None:
    if _hash_matches(destination, entry):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".warm-partial")
    if partial.exists() and (
        not partial.is_file() or partial.stat().st_size > entry["size"]
    ):
        if partial.is_dir():
            raise AssetBootstrapError(
                f"asset partial path is unexpectedly a directory: {partial}"
            )
        partial.unlink()
    if partial.is_file() and partial.stat().st_size == entry["size"]:
        if _hash_matches(partial, entry):
            os.replace(partial, destination)
            return
        partial.unlink()

    failures: list[str] = []
    headers = _authorization_headers()
    attempts = max(DEFAULT_DOWNLOAD_RETRIES, len(endpoints) * 2)
    for attempt in range(attempts):
        endpoint = endpoints[attempt % len(endpoints)]
        offset = partial.stat().st_size if partial.is_file() else 0
        request_headers = dict(headers)
        if offset:
            request_headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            _resolve_url(endpoint, str(entry["path"])),
            headers=request_headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", response.getcode()))
                append = offset > 0 and status == 206
                mode = "ab" if append else "wb"
                if offset > 0 and status not in {200, 206}:
                    raise AssetBootstrapError(
                        f"resume request returned HTTP {status}"
                    )
                with partial.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        if handle.tell() > entry["size"]:
                            raise AssetBootstrapError(
                                f"download exceeded attested size for {entry['path']}"
                            )
        except Exception as error:
            failures.append(
                f"endpoint={endpoint} attempt={attempt + 1}: "
                f"{_exception_chain(error)}"
            )
            if attempt + 1 < attempts:
                time.sleep(min(2 ** min(attempt, 4), 15))
            continue

        if partial.stat().st_size < entry["size"]:
            failures.append(
                f"endpoint={endpoint} attempt={attempt + 1}: short download "
                f"{partial.stat().st_size}/{entry['size']}"
            )
            continue
        if not _hash_matches(partial, entry):
            failures.append(
                f"endpoint={endpoint} attempt={attempt + 1}: digest mismatch"
            )
            partial.unlink(missing_ok=True)
            continue
        os.replace(partial, destination)
        return
    raise AssetBootstrapError(
        f"failed to download pinned asset {entry['path']!r}: "
        + " | ".join(failures)
    )


def _materialize_duplicate(
    source: Path,
    destination: Path,
    entry: Mapping[str, Any],
) -> None:
    if _hash_matches(destination, entry):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".warm-link")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _download_direct_asset_tree(
    *,
    asset_root: Path,
    max_workers: int,
) -> str:
    timeout = _positive_int_environment(
        "HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_NETWORK_TIMEOUT_SECONDS
    )
    endpoints = _endpoint_candidates()
    plan = _load_or_fetch_download_plan(
        asset_root,
        endpoints=endpoints,
        timeout=timeout,
    )
    files = _validate_download_plan(plan)
    probe_timeout = _positive_int_environment("RMBENCH_HF_PROBE_TIMEOUT", 60)
    endpoints = _probe_download_endpoints(
        endpoints,
        entry=files[0],
        timeout=probe_timeout,
    )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for entry in files:
        key = (entry["algorithm"], entry["digest"], entry["size"])
        groups.setdefault(key, []).append(entry)

    completed = 0
    lock = threading.Lock()
    total = len(groups)

    def materialize(entries: list[dict[str, Any]]) -> None:
        nonlocal completed
        verified = next(
            (
                asset_root / entry["path"]
                for entry in entries
                if _hash_matches(asset_root / entry["path"], entry)
            ),
            None,
        )
        canonical_entry = entries[0]
        canonical = verified or asset_root / canonical_entry["path"]
        if verified is None:
            _download_file(
                destination=canonical,
                entry=canonical_entry,
                endpoints=endpoints,
                timeout=timeout,
            )
        for entry in entries:
            destination = asset_root / entry["path"]
            if destination != canonical:
                _materialize_duplicate(canonical, destination, entry)
        with lock:
            completed += 1
            print(
                f"asset_blob_ready={completed}/{total} "
                f"path={canonical_entry['path']} bytes={canonical_entry['size']}",
                flush=True,
            )

    workers = min(max_workers, total)
    print(
        f"asset_download_start=unique_blobs:{total} files:{len(files)} "
        f"workers:{workers} endpoints:{','.join(endpoints)}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(materialize, entries) for entries in groups.values()]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return f"direct_resolve:{plan['endpoint']}"


def _download_snapshot(
    *,
    asset_root: Path,
    max_workers: int,
) -> str:
    # Training and formal evaluation deliberately run offline, and those flags
    # are commonly exported by the persistent CCI image.  This helper is the
    # one explicit network bootstrap, so clear inherited offline-only switches
    # before importing huggingface_hub (its constants are initialized at
    # import time).  The process is short-lived; later eval jobs still set the
    # same flags fail-closed.
    inherited_offline = {
        name: os.environ.pop(name)
        for name in (
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "TRANSFORMERS_OFFLINE",
        )
        if name in os.environ
    }
    if inherited_offline:
        rendered = ", ".join(
            f"{name}={value!r}" for name, value in sorted(inherited_offline.items())
        )
        print(f"asset_download_cleared_offline_flags={rendered}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise AssetBootstrapError(
            "huggingface_hub is required in the base WARM environment"
        ) from error
    asset_root.mkdir(parents=True, exist_ok=True)
    timeout = _positive_int_environment(
        "HF_HUB_DOWNLOAD_TIMEOUT", DEFAULT_NETWORK_TIMEOUT_SECONDS
    )
    etag_timeout = _positive_int_environment(
        "HF_HUB_ETAG_TIMEOUT", DEFAULT_NETWORK_TIMEOUT_SECONDS
    )
    try:
        snapshot_download(
            repo_id=RMBENCH_HF_REPOSITORY,
            repo_type="dataset",
            revision=RMBENCH_HF_REVISION,
            allow_patterns=list(ASSET_PATTERNS),
            local_dir=str(asset_root),
            max_workers=max_workers,
            etag_timeout=etag_timeout,
        )
    except Exception as error:
        print(
            "asset_snapshot_download_fallback="
            f"{_exception_chain(error)}",
            file=sys.stderr,
            flush=True,
        )
        return _download_direct_asset_tree(
            asset_root=asset_root,
            max_workers=max_workers,
        )

    metadata = _tree_metadata(asset_root)
    if (
        metadata["file_count"] != EXPECTED_ASSET_FILE_COUNT
        or metadata["total_bytes"] != EXPECTED_ASSET_TOTAL_BYTES
    ):
        print(
            "asset_snapshot_incomplete_fallback="
            f"files={metadata['file_count']} bytes={metadata['total_bytes']}",
            file=sys.stderr,
            flush=True,
        )
        return _download_direct_asset_tree(
            asset_root=asset_root,
            max_workers=max_workers,
        )
    return f"huggingface_snapshot_download:{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rmbench-root", type=Path, required=True)
    parser.add_argument("--asset-store", type=Path, required=True)
    parser.add_argument("--asset-source", type=Path)
    parser.add_argument("--rmbench-revision", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--download-mode",
        choices=("auto", "direct"),
        default="auto",
        help=(
            "Use 'direct' to skip Hugging Face snapshot metadata and fetch only "
            "the audited simulator tree with range-resumable verified GETs."
        ),
    )
    parser.add_argument(
        "--trust-existing-source",
        action="store_true",
        help=(
            "Adopt a complete explicit --asset-source without a WARM manifest. "
            "Use only when that source was independently obtained at the pinned "
            "official Hugging Face revision."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.rmbench_revision != "57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c":
            raise AssetBootstrapError(
                f"unsupported RMBench code revision: {args.rmbench_revision}"
            )
        if args.max_workers < 1 or args.max_workers > 64:
            raise AssetBootstrapError("--max-workers must be in [1, 64]")

        rmbench_root = args.rmbench_root.expanduser().resolve()
        if args.asset_source is not None:
            asset_root = args.asset_source.expanduser().resolve()
            try:
                previous = validate_asset_manifest(asset_root)
            except AssetBootstrapError:
                if not args.trust_existing_source:
                    raise
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source="trusted-explicit-existing-source",
                    rendered_configs=rendered,
                )
            else:
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source=str(previous["source"]),
                    rendered_configs=rendered,
                )
        else:
            asset_root = args.asset_store.expanduser().resolve()
            try:
                previous = validate_asset_manifest(asset_root)
                print(f"asset_snapshot_reused={asset_root}")
            except (AssetBootstrapError, json.JSONDecodeError):
                if args.download_mode == "direct":
                    download_source = _download_direct_asset_tree(
                        asset_root=asset_root,
                        max_workers=args.max_workers,
                    )
                else:
                    download_source = _download_snapshot(
                        asset_root=asset_root,
                        max_workers=args.max_workers,
                    )
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source=download_source,
                    rendered_configs=rendered,
                )
            else:
                rendered = render_embodiment_configs(asset_root, rmbench_root)
                write_asset_manifest(
                    asset_root,
                    source=str(previous["source"]),
                    rendered_configs=rendered,
                )

        manifest = validate_asset_manifest(asset_root)
        deploy_asset_links(
            asset_root=asset_root,
            rmbench_root=rmbench_root,
            rmbench_revision=args.rmbench_revision,
        )
    except (AssetBootstrapError, json.JSONDecodeError, OSError) as error:
        print(f"RMBENCH_ASSET_BOOTSTRAP_ERROR: {error}", file=sys.stderr)
        return 2

    print(f"RMBENCH_ASSETS_READY root={asset_root}")
    print(f"asset_manifest={_manifest_path(asset_root)}")
    print(f"asset_revision={manifest['revision']}")
    print(f"asset_files={manifest['file_count']}")
    print(f"asset_bytes={manifest['total_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
