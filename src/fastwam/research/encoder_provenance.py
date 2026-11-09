"""Qualify an OS provenance difference with exact saved DINO feature parity.

The original numerical runtime mapping is always recorded. Only its platform
string may differ; every numerical field and four fixed factual feature arrays
must match before the original immutable policy worker can run. The legacy
full-map hash receives an explicitly recorded comparison view, never an edited
encoder contract or a fabricated claim about the actual host.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def runtime_differences(expected, actual):
    if set(expected)!=set(actual):raise ValueError('encoder runtime field set differs')
    differences={k:dict(expected=expected[k],actual=actual[k]) for k in expected if expected[k]!=actual[k]}
    strict={k:v for k,v in differences.items() if k!='platform'}
    if strict:raise ValueError('encoder numerical runtime differs: '+json.dumps(strict,sort_keys=True))
    return differences


def require_feature_parity(expected, actual):
    import numpy as np
    if (expected.shape!=(4,768) or actual.shape!=expected.shape or expected.dtype!=np.float32 or
            actual.dtype!=expected.dtype or not np.isfinite(actual).all() or
            expected.tobytes(order='C')!=actual.tobytes(order='C')):
        raise ValueError('exact original factual DINO array parity failed')


def install(plan_path, expected_plan_sha256, output, *, memory_fraction):
    """Run only in the fresh role process before any policy/model construction."""
    import gc,os,random,sys,traceback
    path=Path(plan_path)
    if file_sha(path)!=expected_plan_sha256:raise ValueError('encoder qualification plan changed')
    plan=json.loads(path.read_text())
    if plan['schema']!='warm.encoder-provenance-qualification.v1' or len(plan['goldens'])!=4:
        raise ValueError('four pre-fixed original factual observations required')
    for item in plan['inputs']:
        if file_sha(item['path'])!=item['sha256']:raise ValueError('encoder qualification input changed: '+item['path'])
    pinned={item['path'] for item in plan['inputs']}
    required={plan['encoder_contract'],plan['runtime_args'],plan['online_contract']}
    required.update(p for g in plan['goldens'] for p in (g['observation'],g['features']))
    if not required.issubset(pinned) or len({g['observation'] for g in plan['goldens']})!=4:
        raise ValueError('all four distinct factual observations and feature arrays must be hash-bound')
    expected=json.loads(Path(plan['encoder_contract']).read_text())['runtime']
    receipt=dict(status='checking',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 qualification_plan_sha256=expected_plan_sha256,expected_m1_runtime=expected,
                 exact_feature_checks=[],policy_loaded=False,contract_edited=False,
                 permitted_provenance_field='platform',golden_arrays_are_not_training_targets=True)
    destination=Path(output)
    if destination.exists():raise FileExistsError('qualification receipt already exists')
    def save():
        temporary=destination.with_suffix('.partial')
        temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(destination)
    original=None;cpu_rng=cuda_rng=numpy_rng=python_rng=None;reader=None
    try:
        for name,key in [('CUBLAS_WORKSPACE_CONFIG','cublas_workspace_config'),('NVIDIA_TF32_OVERRIDE','nvidia_tf32_override')]:
            if expected[key] is None:os.environ.pop(name,None)
            else:os.environ[name]=expected[key]
        import numpy as np
        import torch
        if torch.cuda.is_initialized():raise ValueError('numerical policy must be restored before CUDA initialization')
        torch.set_float32_matmul_precision(expected['float32_matmul_precision'])
        for attr in ('allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction'):
            setattr(torch.backends.cuda.matmul,attr,expected['cuda_matmul_'+attr])
        for attr in ('enabled','allow_tf32','benchmark','deterministic'):
            setattr(torch.backends.cudnn,attr,expected['cudnn_'+attr])
        torch.use_deterministic_algorithms(expected['deterministic_algorithms'],warn_only=bool(expected.get('deterministic_warn_only')))
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        from fastwam.memory import runtime_fingerprint
        if not Path(runtime_fingerprint.__file__).resolve().is_relative_to(Path(plan['core_code']).resolve()/'src'):
            raise ValueError('qualification must use the original immutable core runtime')
        original=runtime_fingerprint.current_encoder_runtime
        actual=original('cuda');receipt['actual_encoder_runtime']=actual
        receipt['numerical_policy_restored_from_m1']=True
        differences=runtime_differences(expected,actual);receipt['provenance_differences']=differences
        receipt['legacy_comparison_view_applied']=bool(differences)
        save()
        def comparison_view(device):
            observed=original(device)
            runtime_differences(expected,observed)
            if observed!=actual:raise ValueError('encoder runtime changed since actual host qualification')
            return dict(expected)
        if differences:runtime_fingerprint.current_encoder_runtime=comparison_view
        # Append only research helpers missing from the bound core. Existing
        # model, processor and inference modules continue to resolve to core.
        import fastwam.research
        helper_directory=str(Path(__file__).resolve().parent)
        if helper_directory not in fastwam.research.__path__:fastwam.research.__path__.append(helper_directory)
        from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
        runtime=json.loads(Path(plan['runtime_args']).read_text());contract=json.loads(Path(plan['online_contract']).read_text())
        cpu_rng=torch.get_rng_state().clone();cuda_rng=[s.clone() for s in torch.cuda.get_rng_state_all()]
        numpy_rng=np.random.get_state();python_rng=random.getstate()
        reader=FrozenFactualEffectReader.from_source_runtime(runtime,contract)
        receipt['reader']=reader.receipt
        for golden in plan['goldens']:
            with np.load(golden['observation'],allow_pickle=False) as data:
                obs={'observation':{key:{'rgb':data[key].copy()} for key in ('head_camera','left_camera','right_camera')},
                     'joint_action':{'vector':data['native_qpos'].copy()}}
            expected_array=np.load(golden['features'],allow_pickle=False)
            actual_array=reader.encode(obs);require_feature_parity(expected_array,actual_array)
            receipt['exact_feature_checks'].append(dict(observation=golden['observation'],features=golden['features'],
                                                       dtype='float32',shape=[4,768],exact=True,
                                                       array_bytes_sha256=hashlib.sha256(actual_array.tobytes(order='C')).hexdigest()))
            save()
        receipt.update(status='qualified_exact_factual_features_and_numerical_runtime',
                       qualified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       scope='Compatibility qualification only; zero new policy actions, no efficacy claim.')
        save()
        return receipt
    except BaseException as exc:
        if original is not None:
            from fastwam.memory import runtime_fingerprint
            runtime_fingerprint.current_encoder_runtime=original
        receipt.update(status='failed_before_policy',error=repr(exc),traceback=traceback.format_exc());save()
        raise
    finally:
        if cpu_rng is not None:
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
            np.random.set_state(numpy_rng);random.setstate(python_rng)
        reader=None;gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():sys.modules['torch'].cuda.empty_cache()
