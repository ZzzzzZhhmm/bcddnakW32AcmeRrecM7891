import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/probe_nonreal_full_null.py"
spec = importlib.util.spec_from_file_location("full_null_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_prefixes_are_frozen_per_episode_and_leave_complete_horizon():
    records = [(SimpleNamespace(dataset_index=0, episode_index=e, frame_index=f), task)
               for task, e in (("a", 1), ("a", 2), ("b", 3), ("b", 4)) for f in range(100)]
    rows = probe.select_prefixes(records, ["a", "b"], 2, 5)
    assert len(rows) == 20
    assert len({(r['episode_index'], r['frame_index']) for r in rows}) == 20
    assert all(r['frame_index'] % 4 == 0 and r['frame_index'] + 32 < 100 for r in rows)
    with pytest.raises(ValueError, match="insufficient DEV"):
        probe.select_prefixes(records, ["a"], 3, 1)


def test_factual_context_excludes_all_teacher_fields_and_keeps_history():
    from dataclasses import replace
    torch = pytest.importorskip("torch")
    from fastwam.models.warm import retrospection_model as rm
    from tests.test_warm_retrospection_model_torch import _source_context
    ctx = _source_context(with_teachers=False)
    ctx = replace(ctx,
                  candidate_normalized_phase=torch.zeros_like(ctx.candidate_support),
                  candidate_event_ordinal=torch.zeros_like(ctx.candidate_support, dtype=torch.long),
                  episode_role_ids=torch.zeros_like(ctx.episode_mask, dtype=torch.long),
                  episode_relative_age=torch.zeros_like(ctx.episode_mask, dtype=torch.float32),
                  episode_action_relative_age=torch.zeros_like(ctx.episode_action_mask, dtype=torch.float32))
    # Supply only factual fields. Accessing targets would be a KeyError.
    names = {
        'query_context':'WARM_CURRENT_CONTEXT', 'candidate_context':'WARM_CANDIDATE_CONTEXT',
        'candidate_actions':'WARM_CANDIDATE_MU', 'candidate_start_proprio':'WARM_CANDIDATE_START_PROPRIO',
        'candidate_effect_pre':'WARM_CANDIDATE_EFFECT_PRE', 'candidate_effect_delta':'WARM_CANDIDATE_EFFECT_DELTA',
        'candidate_timing':'WARM_CANDIDATE_TIMING', 'candidate_support':'WARM_CANDIDATE_SUPPORT',
        'candidate_valid_mask':'WARM_CANDIDATE_MASK', 'candidate_normalized_phase':'WARM_CANDIDATE_NORMALIZED_PHASE',
        'candidate_event_ordinal':'WARM_CANDIDATE_EVENT_ORDINAL', 'episode_tokens':'WARM_EPISODE_TOKENS',
        'episode_mask':'WARM_EPISODE_MASK', 'episode_action_summaries':'WARM_EPISODE_ACTION_SUMMARIES',
        'episode_action_mask':'WARM_EPISODE_ACTION_MASK', 'episode_role_ids':'WARM_EPISODE_ROLE_IDS',
        'episode_relative_age':'WARM_EPISODE_RELATIVE_AGE', 'episode_action_relative_age':'WARM_EPISODE_ACTION_RELATIVE_AGE',
    }
    sample = {getattr(rm, constant): getattr(ctx, field)[0] for field, constant in names.items()}
    result = probe.factual_context(sample, device="cpu", dtype=torch.float32)
    assert result.target_action is None and result.target_effect is None
    assert result.current_semantic_teacher is None and result.future_valid_mask is None
    assert torch.equal(result.episode_tokens, ctx.episode_tokens[:1])
    changed = probe.perturb_candidates(result)
    assert not torch.equal(result.candidate_actions, changed.candidate_actions)
    assert torch.equal(result.candidate_valid_mask, changed.candidate_valid_mask)
    assert torch.equal(result.episode_action_summaries, changed.episode_action_summaries)
    assert not changed.candidate_actions[~result.candidate_valid_mask].any()


def test_source_diagnosis_rejects_conditioning_or_gate_changes():
    torch = pytest.importorskip("torch")
    arrays = {key: torch.ones(2) for key in ("base_gaussian", "conditioning", "g", "alpha", "selected_index", "adapted_actions", "valid", "source", "source_noise_scale")}
    reference = {"arrays": arrays}
    alternative = {"arrays": {key: value.clone() for key,value in arrays.items()}}
    alternative['arrays']['source'] *= 2
    assert probe.source_pair_metrics(reference, alternative)['source_rms_delta'] == 1
    alternative['arrays']['conditioning'] *= 0
    with pytest.raises(ValueError, match='conditioning'):
        probe.source_pair_metrics(reference, alternative)


@pytest.mark.parametrize('mode', ['full', 'scale_only', 'gaussian'])
def test_source_collection_respects_real_model_lifetime_lock(tmp_path, mode):
    torch = pytest.importorskip('torch')
    from tests.test_nonreal_probe_torch import twin, resolve
    from tests.test_warm_retrospection_model_torch import _source_context
    model = twin()
    model.configure_research_probe(source_mode=mode, capture=True)
    calls = []
    def infer(context):
        resolve(model, context)
        calls.append(mode)
        return torch.zeros(32,14), .1, model._last_research_probe
    rows = probe.source_query(_source_context(with_teachers=False), infer, tmp_path,
                              'prefix-0000', {}, {'task':'test'}, mode)
    assert calls == ([mode,mode] if mode == 'full' else [mode])
    assert model._warm_online_experiment_locked
    assert model._warm_research_probe.source_mode == mode
    assert len(rows) == 1 and rows[0]['mode'] == mode
