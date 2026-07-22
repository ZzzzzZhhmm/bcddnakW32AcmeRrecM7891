from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from fastwam.datasets.rmbench.constants import (
    CONVERSION_SCHEMA,
    CONVERSION_SCHEMA_VERSION,
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_RMBENCH_TASKS,
    OFFICIAL_TASK_CONFIG,
)
from fastwam.datasets.rmbench.source import canonical_json_bytes
from scripts.validate_rmbench_conversion import (
    RMBenchConversionValidationError,
    validate_conversion,
)


DATA_REVISION = "8" * 40
CODE_REVISION = "5" * 40


def _artifact(root: Path) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    catalog = b'{"catalog":"test"}\n'
    (meta / "warm_episode_catalog.json").write_bytes(catalog)
    artifacts = [
        {
            "path": "meta/warm_episode_catalog.json",
            "size": len(catalog),
            "sha256": sha256(catalog).hexdigest(),
        }
    ]
    manifest = {
        "schema": CONVERSION_SCHEMA,
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "source": {
            "dataset": "TianxingChen/RMBench",
            "revision": DATA_REVISION,
            "rmbench_code_revision": CODE_REVISION,
            "task_config": OFFICIAL_TASK_CONFIG,
            "source_tree_sha256": "1" * 64,
        },
        "output": {
            "dataset_id": "fixture",
            "data_revision": "fixture-v1",
            "lerobot_codebase_version": "v2.1",
            "catalog_relpath": "meta/warm_episode_catalog.json",
            "catalog_sha256": sha256(catalog).hexdigest(),
            "artifact_tree_sha256": sha256(canonical_json_bytes(artifacts)).hexdigest(),
            "artifacts": artifacts,
        },
        "protocol": {
            "data_profile": "official50-dev45",
            "official_task_allow_list": list(OFFICIAL_RMBENCH_TASKS),
            "episodes_per_task": OFFICIAL_EPISODES_PER_TASK,
            "split": {"dev_per_task": 5},
        },
        "episodes": [
            {"fixture": index}
            for index in range(len(OFFICIAL_RMBENCH_TASKS) * OFFICIAL_EPISODES_PER_TASK)
        ],
    }
    manifest["manifest_sha256"] = sha256(canonical_json_bytes(manifest)).hexdigest()
    (meta / "rmbench_conversion_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_conversion_validator_binds_revision_catalog_and_artifact_bytes(tmp_path: Path) -> None:
    _artifact(tmp_path)
    value = validate_conversion(
        tmp_path,
        source_revision=DATA_REVISION,
        rmbench_code_revision=CODE_REVISION,
        hash_artifacts=True,
    )
    assert value["source"]["revision"] == DATA_REVISION

    (tmp_path / "meta" / "warm_episode_catalog.json").write_bytes(b"corrupt")
    with pytest.raises(RMBenchConversionValidationError, match="catalog SHA-256"):
        validate_conversion(
            tmp_path,
            source_revision=DATA_REVISION,
            rmbench_code_revision=CODE_REVISION,
            hash_artifacts=True,
        )


def test_conversion_validator_rejects_wrong_source_revision(tmp_path: Path) -> None:
    _artifact(tmp_path)
    with pytest.raises(RMBenchConversionValidationError, match="dataset revision"):
        validate_conversion(
            tmp_path,
            source_revision="9" * 40,
            rmbench_code_revision=CODE_REVISION,
            hash_artifacts=False,
        )
