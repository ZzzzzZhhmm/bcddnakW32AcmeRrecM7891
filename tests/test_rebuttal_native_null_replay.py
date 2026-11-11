"""Reject false same-context qualifications before any native counterfactual."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('native_null_rules', Path(__file__).resolve().parents[1] / 'src/fastwam/research/native_null_replay.py')
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)


@dataclass
class Context:
    query_context: object
    candidate_valid_mask: object
    target_effect: object = None


def source():
    noise = torch.ones(1, 32, 14, dtype=torch.bfloat16)
    conditioning = torch.ones(1, 12, 4096, dtype=torch.bfloat16)
    arrays = dict(base_gaussian=noise, source=noise * 2, conditioning=conditioning,
                  g=torch.tensor([.2]), alpha=torch.tensor([.2]), selected_index=torch.tensor([3]),
                  predicted_effect=torch.zeros(1, 32, 3), historical_effect=torch.ones(1, 32, 3), required_effect=torch.ones(1, 3))
    return dict(force_null=False, source_mode='full', arrays=arrays)


def test_asset_patterns_use_original_root_and_reject_foreign_artifacts(tmp_path):
    root = tmp_path / 'WARM'
    paths = dict(warm_base_checkpoint_path=root / 'checkpoints/fastwam_release/base.pt',
                 warm_vae_checkpoint_path=root / 'checkpoints/vae.pt',
                 warm_text_encoder_path=root / 'checkpoints/text.pt',
                 warm_tokenizer_path=root / 'checkpoints/tokenizer')
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    runtime = {key: str(path) for key, path in paths.items()}
    assert rules.asset_working_directory(runtime) == root
    other = tmp_path / 'unbound.pt'
    other.touch()
    with pytest.raises(ValueError, match='outside'):
        rules.asset_working_directory(dict(runtime, warm_text_encoder_path=str(other)))


def test_loaded_asset_metadata_matches_bound_assets_and_factory_is_restored(tmp_path):
    from types import SimpleNamespace
    root = tmp_path / 'WARM'
    paths = dict(warm_base_checkpoint_path=root / 'checkpoints/fastwam_release/base.pt',
                 warm_vae_checkpoint_path=root / 'checkpoints/vae.pt',
                 warm_text_encoder_path=root / 'checkpoints/text.pt',
                 warm_tokenizer_path=root / 'checkpoints/tokenizer')
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    runtime = {key: str(path) for key, path in paths.items()}
    original = dict(vae='checkpoints/vae.pt', text_encoder='checkpoints/text.pt',
                    tokenizer='checkpoints/tokenizer', video_dit='untouched')
    model = SimpleNamespace(model_paths=dict(original))
    factory = lambda *_args, **_kwargs: model
    deployment = SimpleNamespace(instantiate=factory)
    records = []
    with rules.bound_asset_metadata(deployment, runtime, records.append):
        assert deployment.instantiate() is model
        assert model.model_paths['vae'] == str(paths['warm_vae_checkpoint_path'].resolve())
        assert model.model_paths['video_dit'] == 'untouched'
    assert deployment.instantiate is factory and len(records) == 1
    model.model_paths = dict(original, vae='checkpoints/text.pt')
    with pytest.raises(ValueError, match='contradicts'):
        with rules.bound_asset_metadata(deployment, runtime, records.append):
            deployment.instantiate()
    assert deployment.instantiate is factory
    assert model.model_paths['text_encoder'] == original['text_encoder']
    with pytest.raises(ValueError, match='exactly one'):
        with rules.bound_asset_metadata(deployment, runtime, records.append):
            pass
    assert deployment.instantiate is factory


def test_restore_rejects_future_unknown_or_missing_witness_and_clones_tensors():
    payload = dict(query_context=torch.ones(1, 3), candidate_valid_mask=torch.ones(1, 32, dtype=torch.bool),
                   actual_base_gaussian=torch.ones(1), actual_source=torch.ones(1), actual_conditioning_tokens=torch.ones(1))
    context = rules.restore_factual_context(payload, Context, device='cpu')
    assert context.query_context.data_ptr() != payload['query_context'].data_ptr()
    assert context.candidate_valid_mask.dtype == torch.bool
    for key in ['target_effect', 'unknown']:
        with pytest.raises(ValueError, match='future'):
            rules.restore_factual_context(dict(payload, **{key: torch.ones(1)}), Context, device='cpu')
    with pytest.raises(ValueError, match='incomplete'):
        rules.restore_factual_context({k: v for k, v in payload.items() if k != 'actual_source'}, Context, device='cpu')


def test_full_replay_rejects_changed_action_and_changed_actual_conditioning():
    p = source()
    action = np.zeros((32, 14), dtype=np.float32)
    witnesses = {key: p['arrays'][name].clone() for key, name in [('actual_base_gaussian', 'base_gaussian'), ('actual_source', 'source'), ('actual_conditioning_tokens', 'conditioning')]}
    rules.require_reproduced_full(action, action[None], p, p, witnesses)
    changed = action.copy()
    changed[0, 0] = .001
    with pytest.raises(ValueError, match='H32'):
        rules.require_reproduced_full(changed, action, p, p, witnesses)
    witnesses['actual_conditioning_tokens'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='witness'):
        rules.require_reproduced_full(action, action, p, p, witnesses)


def test_explicit_null_requires_Gaussian_zero_gates_and_zero_candidate_slots():
    original = source()
    null = source()
    null['force_null'] = True
    a = null['arrays']
    a['source'] = a['base_gaussian'].clone()
    a['g'] = torch.zeros(1)
    a['source_gate'] = torch.zeros(1)
    a['conditioning'][:, 8:] = 0
    assert rules.require_explicit_null_probe(null, original)['natural_skill_absence_certified'] is False
    a['conditioning'][0, 8, 0] = 1
    with pytest.raises(ValueError, match='four candidate slots'):
        rules.require_explicit_null_probe(null, original)
    a['conditioning'][:, 8:] = 0
    a['source_gate'][0] = .01
    with pytest.raises(ValueError, match='gates'):
        rules.require_explicit_null_probe(null, original)
