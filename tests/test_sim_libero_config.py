from __future__ import annotations

from pathlib import Path

import yaml


def test_sim_libero_warm_online_fields_resolve_under_one_mapping() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "sim_libero.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    online = value["EVALUATION"]["warm_online"]
    assert set(online) == {
        "enabled",
        "contract_path",
        "pair_contract_path",
        "parity_report_path",
        "m1_data_config_path",
        "training_attestation_path",
        "training_run_contract_path",
        "validation_run_contract_path",
        "base_checkpoint_path",
        "bank_directory",
        "normalizer_contract_path",
        "encoder_contract_path",
        "camera_contract_path",
        "dino_checkpoint_path",
        "catalog_path",
        "audit_report_path",
        "evaluation_namespace",
        "top_k",
        "dino_device",
        "dino_batch_size",
    }
    assert online["training_attestation_path"] is None

