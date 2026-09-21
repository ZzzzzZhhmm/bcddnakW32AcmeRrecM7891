import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_keeps_schedule_and_parent_untouched():
    module=load_script('nonreal_resume')
    parent=dict(max_steps=30000,run_steps=300,batch_size=8,gradient_accumulation_steps=4,
                output_dir='original',resume=None,model=dict(source_policy='fixed_context_top1'))
    child=module.continuation_config(parent,state='/old/step_000300',output='/new',parent_step=300,target_step=15000)
    assert child['max_steps']==30000
    assert child['run_steps']==14700
    assert child['batch_size']*child['gradient_accumulation_steps']*4==128
    assert parent['resume'] is None and parent['output_dir']=='original'
    for target in (300,30001):
        with pytest.raises(ValueError,match='target must advance'):
            module.continuation_config(parent,state='/old',output='/new',parent_step=300,target_step=target)


def test_successful_process_without_target_checkpoint_is_incomplete(tmp_path):
    module=load_script('nonreal_resume')
    training=tmp_path/'training'
    training.mkdir()
    row=dict(schema='warm.training-metrics',version=1,stage='shared',step=1000,
             loss=.5,grad_norm=.1,learning_rate=1e-5,steps_per_second=.2,metrics={})
    (training/'training_metrics.jsonl').write_text(json.dumps(row)+'\n')
    plan=dict(training_output=str(training),config=str(tmp_path/'resume_train.yaml'),
              parent_step=300,target_step=1000,source_sha256='a'*64,config_sha256='b'*64)
    report=module.summarize(plan,exit_code=0)
    assert report['status']=='incomplete'
    assert (tmp_path/'EXPERIMENT_RECORD.generated.md').is_file()


def test_profile_excludes_warmup_and_rejects_duplicate_or_missing_queries():
    module=load_script('profile_nonreal_online_replay')
    def row(frame,seconds,warmup=False):
        return dict(dataset_index=0,recorded_episode=5,recorded_frame=frame,warmup=warmup,
                    end_to_end_s=seconds,peak_allocated_gib=12,peak_reserved_gib=14)
    rows=[row(0,100,True),row(0,1),row(4,3)]
    report=module.summarize(rows,expected_count=2)
    assert report['median_s']==2
    assert report['warmup_queries']==1
    assert report['episodes']==1
    with pytest.raises(ValueError,match='incomplete'):
        module.summarize(rows,expected_count=3)
    with pytest.raises(ValueError,match='duplicate'):
        module.summarize([row(0,1),row(0,2)],expected_count=2)
