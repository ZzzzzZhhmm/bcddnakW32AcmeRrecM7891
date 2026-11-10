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


def test_source_only_observer_delegates_without_legacy_probe_or_mode_change():
    from dataclasses import dataclass
    from types import SimpleNamespace
    import torch
    from scripts.qualify_rebuttal_pooled_online import capture_native_replan
    @dataclass
    class Context:
        candidate_actions: torch.Tensor
        target_action: object=None
        target_effect: object=None
        current_semantic_teacher: object=None
        future_valid_mask: object=None
        target_action_valid_mask: object=None
    class Model:
        mode='source_only_no_consequence'
        def _resolve_action_source(self,**kw):
            self.received=kw
            return SimpleNamespace(source=kw['base_gaussian']+1,conditioning_tokens=torch.zeros(1,12,4))
        def infer_action(self,**kw):
            self._resolve_action_source(base_gaussian=kw['base_gaussian'],action_source_context=kw['context'])
            self.original_output={'action':torch.zeros(32,14)}
            return self.original_output
    model=Model();policy=SimpleNamespace(model=model);gaussian=torch.randn(1,32,14);ctx=Context(torch.ones(1,32,32,14))
    with capture_native_replan(policy) as captured:
        value=model.infer_action(online_step='fixed',base_gaussian=gaussian,context=ctx)
    assert value is model.original_output and model.received['base_gaussian'] is gaussian
    assert torch.equal(captured['base_gaussian'],gaussian) and captured['source_calls']==1 and captured['infer_calls']==1
    assert model.mode=='source_only_no_consequence' and not hasattr(model,'_warm_research_probe')
    assert 'infer_action' not in vars(model) and '_resolve_action_source' not in vars(model)
