"""Read-only, CPU-only pooler intake with the original specialist model imports.

This checks a small module interface, not a trained policy or task performance.
The original timing encoding, data recipe and sealed context are not rewritten.
"""
from pathlib import Path
import argparse,ast,hashlib,importlib,json,os,runpy,sys,types
from dataclasses import replace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-code',type=Path,required=True)
    p.add_argument('--preflight-code',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--sealed-context',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();original=a.original_code.resolve();current=a.preflight_code.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU intake requires explicitly hidden CUDA devices')
    if any(k=='fastwam' or k.startswith('fastwam.') for k in sys.modules):raise ValueError('Import original model before any current fastwam package')
    if original==current or not (original/'src/fastwam/models/warm/retrospection_config.py').is_file():raise ValueError('Distinct existing original source is required')
    if any(a.output.with_suffix(suffix).exists() for suffix in ['.json', '.binding.json', '.dataset_audit.json', '.failure.json']):
        raise ValueError('Preserve every previous intake result and failure receipt')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    before={str(path):sha(path) for path in [a.recipe,a.sealed_context]}
    sys.path[:0]=[str(original/'src'),str(original),str(current)]
    import fastwam
    if not Path(fastwam.__file__).resolve().is_relative_to(original):raise ValueError('Original fastwam package binding failed')
    # Only research helpers absent from the legacy checkout may fall back.
    fastwam.__path__.append(str(current/'src/fastwam'))
    scripts=importlib.import_module('scripts')
    if hasattr(scripts,'__path__'):scripts.__path__.append(str(current/'scripts'))
    from fastwam.models.warm import retrospection_config
    from fastwam import runtime
    for module in [retrospection_config,runtime]:
        if not Path(module.__file__).resolve().is_relative_to(original):raise ValueError('A model/runtime import escaped the original source')
    import torch
    from omegaconf import OmegaConf
    recipe=OmegaConf.load(a.recipe);cfg=retrospection_config.WarmRetrospectionConfig.from_dict(OmegaConf.to_container(recipe.model.retrospection,resolve=True))
    # The original recipes encode two timing facts per canonical gripper.
    # Read the actual recipe; never change it to satisfy a newer contract.
    if cfg.timing_dim != 2 * len(cfg.canonical_gripper_dims):
        raise ValueError('Original per-gripper timing contract does not match the recipe')
    if torch.cuda.is_initialized():raise ValueError('CPU intake initialized CUDA')
    # Execute only the two unchanged small-module class definitions. Importing
    # current policy classes would require a SourceAcceptance API absent from
    # the original checkpoint source. No policy shim or model substitution is
    # permitted; this intake deliberately does not qualify a full policy.
    from fastwam.models.warm.retrospective_event_adapter import (
        RetrospectiveEventAdapter, RetrospectiveEventAdapterOutput,
    )
    importlib.import_module('fastwam.research')
    pool_path = current / 'src/fastwam/research/pooled_memory_prior.py'
    simple_path = current / 'src/fastwam/research/simple_prior.py'
    def definition(path, name):
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name == name]
        if len(nodes) != 1: raise ValueError('Expected one unchanged class: ' + name)
        return nodes[0]
    module = types.ModuleType('fastwam.research.pooled_memory_prior')
    module.__dict__.update(torch=torch, nn=torch.nn, replace=replace,
        RetrospectiveEventAdapter=RetrospectiveEventAdapter,
        RetrospectiveEventAdapterOutput=RetrospectiveEventAdapterOutput,
        SPATIAL_TOKENS=4, PHASE_DIM=4,
        POOL_ARCHITECTURE='all-events-linear-four-query-attention-v1')
    for path, name in [(simple_path, 'DeterministicEvent'), (pool_path, 'PooledMemoryEvent')]:
        node = definition(path, name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), module.__dict__)
    sys.modules[module.__name__] = module
    torch.set_num_threads(4)
    sys.argv=[str(current/'scripts/preflight_rebuttal_pooled_memory.py'),'--recipe',str(a.recipe),'--sealed-context',str(a.sealed_context),'--training-index','0','--output',str(a.output),'--device','cpu']
    original_build = runtime.build_datasets
    required = {'warm_candidate_mu', 'warm_candidate_mask',
        'warm_candidate_effect_pre', 'warm_candidate_effect_delta',
        'warm_candidate_context', 'warm_candidate_timing',
        'warm_candidate_start_proprio', 'warm_current_context',
        'warm_candidate_normalized_phase', 'warm_candidate_event_ordinal', 'proprio'}
    def audited_build(*args, **kwargs):
        dataset, validation = original_build(*args, **kwargs)
        class AuditedDataset:
            def __getitem__(self, index):
                sample = dataset[index]
                fields = {}
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        fields[key] = dict(shape=list(value.shape), dtype=str(value.dtype))
                    else: fields[key] = dict(type=type(value).__name__)
                missing = sorted(required - set(sample))
                audit = dict(fixed_training_index=index, original_dataset_type=type(dataset).__name__,
                    sample_fields=fields, required_fields=sorted(required), missing_fields=missing,
                    status='incompatible_original_dataset' if missing else 'required_fields_present',
                    original_timing_dim=cfg.timing_dim, input_hashes=before,
                    CUDA_initialized=torch.cuda.is_initialized(), optimizer_updates=0,
                    policy_actions=0, original_sample_unchanged=True,
                    scope='Actual original train sample interface only;missing fields are never synthesized')
                a.output.with_suffix('.dataset_audit.json').write_text(json.dumps(audit, indent=2))
                return sample
        return AuditedDataset(), validation
    runtime.build_datasets = audited_build
    try:
        runpy.run_path(str(current/'scripts/preflight_rebuttal_pooled_memory.py'),run_name='__main__')
    except Exception as exc:
        assert all(sha(path)==value for path,value in before.items())
        failure = dict(status='not_qualified', exception_type=type(exc).__name__,
            exception=str(exc), input_hashes=before, CUDA_initialized=torch.cuda.is_initialized(),
            optimizer_updates=0, policy_actions=0,
            scope='Engineering intake failure;not an efficacy or original benchmark result')
        a.output.with_suffix('.failure.json').write_text(json.dumps(failure, indent=2))
        raise
    assert all(sha(path)==value for path,value in before.items()) and not torch.cuda.is_initialized()
    binding=dict(original_code=str(original),research_code=str(current),original_config_module=retrospection_config.__file__,original_runtime_module=runtime.__file__,timing_dim=cfg.timing_dim,pooler_source_hashes={str(p):sha(p) for p in [pool_path,simple_path]},pooler_binding='Exact unchanged class AST only;full current policy classes intentionally not imported',input_hashes=before,CUDA_initialized=False,optimizer_updates=0,policy_actions=0,scope='Small CPU module interface only;not full policy/backbone/training or performance qualification')
    a.output.with_suffix('.binding.json').write_text(json.dumps(binding,indent=2));print(json.dumps(binding))


if __name__=='__main__':main()
