from copy import deepcopy
import pytest
from scripts.prepare_rebuttal_control_training import make_config


def recipe():
    return dict(batch_size=8,gradient_accumulation_steps=4,max_steps=30000,
        model={'_target_':'original','retrospection':dict(loss_retrieval=1.,loss_effect=1.,loss_gate=1.,loss_adaptation=1.)},
        data={'train':{'instruction_variant_seed':1}})


def test_conditioned_control_matches_full_data_optimizer_geometry_and_fixed_recipe():
    original=recipe();saved=deepcopy(original)
    full=make_config(original,kind='full',output='/tmp/full',seed=3407,segment_updates=100)
    control=make_config(original,kind='conditioned_prior',output='/tmp/control',seed=3407,segment_updates=100)
    assert original==saved
    for name in ('data','max_steps','batch_size','gradient_accumulation_steps','seed','run_steps','eval_every','save_every'):
        assert full[name]==control[name]
    assert full['gradient_accumulation_steps']*full['batch_size']==128
    assert control['model']['_target_']=='fastwam.research.simple_prior.create_conditioned_prior'
    assert all(control['model']['retrospection'][name]==0 for name in
        ('loss_retrieval','loss_effect','loss_gate','loss_adaptation'))


def test_training_segments_cannot_silently_change_seed_or_scheduler_budget():
    with pytest.raises(ValueError):make_config(recipe(),kind='conditioned_prior',output='/tmp/a',seed=1,segment_updates=100)
    original=recipe();original['max_steps']=3000
    with pytest.raises(ValueError,match='recipe'):make_config(original,kind='full',output='/tmp/a',seed=3407,segment_updates=100)


def test_pooled_memory_control_starts_a_new_lineage_with_the_common_fixed_recipe():
    full=make_config(recipe(),kind='full',output='/tmp/full',seed=3407,segment_updates=100)
    pooled=make_config(recipe(),kind='pooled_memory_prior',output='/tmp/pooled',seed=3407,segment_updates=100)
    assert pooled['model']['_target_']=='fastwam.research.pooled_memory_prior.create_pooled_memory_prior'
    for name in ('data','max_steps','batch_size','gradient_accumulation_steps','seed','run_steps','eval_every','save_every'):
        assert full[name]==pooled[name]
    assert pooled['resume'] is None and pooled['initialization_checkpoint'] is None
