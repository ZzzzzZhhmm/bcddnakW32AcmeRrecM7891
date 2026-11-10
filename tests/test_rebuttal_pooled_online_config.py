from pathlib import Path
from hydra import compose,initialize_config_dir

CONFIGS=str(Path(__file__).resolve().parents[1]/'configs')

def test_pooled_native_profile_has_its_independently_trained_constructor():
    with initialize_config_dir(version_base='1.3',config_dir=CONFIGS):
        cfg=compose(config_name='sim_rmbench_pooled_memory',overrides=['task=rmbench_warm_online_3cam384_full'])
    assert cfg.model._target_=='fastwam.research.pooled_memory_prior.create_pooled_memory_prior'
    assert cfg.model.source_policy=='fixed_context_top1'
    assert cfg.model.retrospection.action_horizon==32 and cfg.model.retrospection.timing_dim==8
    assert all(cfg.model.retrospection[k]==0 for k in ('loss_retrieval','loss_effect','loss_gate','loss_adaptation'))
    assert cfg.model.retrospection.loss_gist>0 and cfg.model.retrospection.loss_bridge>0

def test_full_native_profile_keeps_its_original_constructor_and_losses():
    with initialize_config_dir(version_base='1.3',config_dir=CONFIGS):
        cfg=compose(config_name='sim_rmbench',overrides=['task=rmbench_warm_online_3cam384_full'])
    assert cfg.model._target_=='fastwam.runtime.create_warm_retrospection'
    assert all(cfg.model.retrospection[k]>0 for k in ('loss_retrieval','loss_effect','loss_gate','loss_adaptation'))


def test_native_recorded_task_instruction_field():
    from scripts.qualify_rebuttal_pooled_online import recorded_instruction
    assert recorded_instruction({'task':'press the button'})=='press the button'

def test_foreign_or_missing_instruction_field_is_rejected():
    import pytest
    from scripts.qualify_rebuttal_pooled_online import recorded_instruction
    for sample in [{'text':'not the native field'},{'task':None},{'task':''}]:
        with pytest.raises(ValueError,match='task instruction'):recorded_instruction(sample)
