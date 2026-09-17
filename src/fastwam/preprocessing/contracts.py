"""Small, CPU-only boundaries shared by benchmark and teleoperation adapters."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

if TYPE_CHECKING:
    import numpy as np


class PreparationError(ValueError):
    pass


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise PreparationError(f"Non-finite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root.resolve() or not path.is_relative_to(root.resolve()):
        raise PreparationError(f"Source path must be inside its root: {relative}")
    return path


def required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise PreparationError(f"{name} must be a nonempty, trimmed string")
    if value.upper().startswith(("TODO", "UNCONFIRMED")) or "待确认" in value:
        raise PreparationError(f"{name} is still unconfirmed")
    return value


@dataclass
class RawEpisode:
    """One complete source episode, already aligned without action padding.

    Rows are command-bearing observations: states[i], images[i], actions[i].
    The shared memory reader uses rows[:-1] actions with rows[1:] consequences.
    The final command stays available for training; terminal raw data stays at
    its source and is referenced by the conversion manifest.
    """
    source_id: str
    task: str
    instruction: str
    split: str
    states: np.ndarray
    actions: np.ndarray
    images: Mapping[str, np.ndarray]
    source_files: tuple[Path, ...]
    annotations: dict[str, Any]

    def validate(self, profile: Mapping[str, Any]) -> None:
        import numpy as np
        for name in ("source_id", "task", "instruction"):
            required_text(getattr(self, name), name)
        if self.split not in {"train", "dev", "test"}:
            raise PreparationError("Every episode must have an explicit train/dev/test split")
        n = len(self.states)
        if n < 2:
            raise PreparationError("At least two command-bearing observations are required")
        for name, dim in (("actions", profile["action_dim"]), ("states", profile["state_dim"])):
            array = np.asarray(getattr(self, name), dtype=np.float32)
            if array.shape != (n, dim) or not np.isfinite(array).all():
                raise PreparationError(f"{name}: expected finite [{n},{dim}], got {array.shape}")
            setattr(self, name, np.ascontiguousarray(array))
        if list(self.images) != profile["camera_keys"]:
            raise PreparationError("Source camera order differs from the declared profile")
        for name, frames in self.images.items():
            if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != n
                    or frames.shape[-1] != 3 or min(frames.shape[1:3]) < 2):
                raise PreparationError(f"{name}: expected uint8 RGB [N,H,W,3]")
            if frames.shape[1] % 2 or frames.shape[2] % 2:
                raise PreparationError("H.264 source image dimensions must be even; no implicit crop")

    def content_sha256(self) -> str:
        """Identity-independent content proof, including raw RGB before encoding."""
        import numpy as np
        digest = sha256(b"warm.raw-episode.v1\0")
        for name, array in (("states", self.states), ("actions", self.actions), *self.images.items()):
            digest.update(name.encode())
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()


class EpisodeAdapter(Protocol):
    def read(self, entry: Mapping[str, Any]) -> RawEpisode: ...


def read_source_manifest(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema") != "warm.source-episodes.v1":
        raise PreparationError("Expected warm.source-episodes.v1 source manifest")
    entries = value.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise PreparationError("Source manifest must explicitly list episodes and splits")
    if any(not isinstance(item, dict) for item in entries):
        raise PreparationError("Every source episode entry must be a JSON object")
    ids = [required_text(item.get("id"), "episode id") for item in entries]
    if len(set(ids)) != len(ids):
        raise PreparationError("Duplicate episode ids in source manifest")
    paths = [required_text(item.get("path"), "episode path") for item in entries]
    if len(set(paths)) != len(paths):
        raise PreparationError("Duplicate episode paths in source manifest")
    if any(item.get("split") not in {"train", "dev", "test"} for item in entries):
        raise PreparationError("Source manifest requires explicit train/dev/test splits")
    return entries
