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


def test_profile_archive_requires_policy_coverage_and_matching_summary(tmp_path):
    module=load_script('verify_nonreal_online_profile')
    data=tmp_path/'profile'
    job=tmp_path/'job_logs'/'job'
    data.mkdir()
    job.mkdir(parents=True)
    def write(path,value):
        path.write_text(json.dumps(value))
    write(job/'run_manifest.json',dict(argv=['--output','/run/profile','--episodes','1',
          '--queries-per-episode','2','--warmup','1'],status='complete',exit_code=0,
          source_sha256='a'*64,source_sha256_after='a'*64,elapsed_s=8))
    write(data/'frozen_episodes.json',[dict(dataset_index=0,recorded_episode=7,start=0)])
    write(data/'runtime_args.json',dict(replan_steps=4,num_inference_steps=10,seed=3407))
    write(data/'manifest.json',dict(replan_steps=4,nfe=10,scope='synthetic test fixture only',
                                   checkpoint_sha256='b'*64,online_contract_sha256='c'*64,task='test'))
    write(data/'data_qualification.json',dict(status='qualified',recorded_prefixes=2))
    rows=[dict(warmup=w,dataset_index=0,recorded_episode=7,recorded_frame=f,
               end_to_end_s=t,peak_allocated_gib=10,peak_reserved_gib=12)
          for w,f,t in [(True,0,100),(False,0,1),(False,4,3)]]
    (data/'timings.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    telemetry=[dict(kind='header',checkpoint_sha256='b'*64,online_run_contract_sha256='c'*64,
                    root_seed=3407,task_name='test')]
    for ep,frames in [(0,[0]),(1,[0,4])]:
        telemetry.append(dict(kind='episode_begin',episode_index=ep))
        telemetry.extend(dict(kind='replan',episode_index=ep,frame_index=f,candidate_count=32,
                              history_before_replan=None,
                              model=dict(source=dict(gate=0,learned_gate=.12,stagnation_score=0))) for f in frames)
        telemetry.append(dict(kind='episode_end',episode_index=ep,
                              reason='recorded_replay_end_no_outcome',executed_policy_actions=len(frames)*4))
    telemetry_path=data/'policy_telemetry.jsonl'
    telemetry_path.write_text('\n'.join(map(json.dumps,telemetry)))
    report=load_script('profile_nonreal_online_replay').summarize(rows,expected_count=2)
    write(data/'summary.json',report)
    assert module.verify(tmp_path,'profile')['median_s']==2
    report['median_s']=999
    write(data/'summary.json',report)
    with pytest.raises(AssertionError,match='median_s'):
        module.verify(tmp_path,'profile')
    report['median_s']=2
    write(data/'summary.json',report)
    telemetry_path.write_text('\n'.join(map(json.dumps,telemetry[:-1])))
    with pytest.raises(AssertionError):
        module.verify(tmp_path,'profile')
