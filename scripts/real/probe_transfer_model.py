#!/usr/bin/env python3
"""Strict checkpoint load and action-only inference on recorded DEV frames.

No SDK, CAN, network robot client or motor command is imported. The result is
an open-loop diagnostic, not closed-loop success or a real-time guarantee.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--nfe', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    if a.samples <= 0 or a.nfe <= 0: raise ValueError('samples and nfe must be positive')
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.utils import misc
    from fastwam.utils.config_resolvers import register_default_resolvers
    from fastwam.preprocessing.contracts import file_sha256
    from fastwam.real.preprocessing.contract import PIPER_JOINT_CONTROL_MODE
    register_default_resolvers()
    cfg = OmegaConf.load(a.config)
    if 'warm_candidates' in cfg.data: raise ValueError('Use the plain transfer config')
    if not torch.cuda.is_available(): raise RuntimeError('Run this model probe in the configured CUDA environment')
    profile_path = Path(str(cfg.data.val.episode_catalog_path)).parent/'config.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))['profile']
    if profile['control_mode'] != PIPER_JOINT_CONTROL_MODE:
        raise ValueError('This probe labels outputs as absolute joint targets; wrong action profile')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(a.output.parent))
    ds = instantiate(cfg.data.val)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device='cuda')
    model.load_checkpoint(str(a.checkpoint), strict_model_state=True,
                          exact_proprio_state=True, allow_legacy_dit=False,
                          forbid_extra_keys=('warm_source',))
    model.eval()
    action_norm = ds.lerobot_dataset.processor.normalizer.normalizers['action']['default']
    rows = []
    with torch.inference_mode():
        for index in range(min(a.samples,len(ds))):
            s = ds[index]
            torch.cuda.synchronize();start=time.perf_counter()
            out = model.infer_action(prompt=None, input_image=s['video'][:,0],
                    action_horizon=32, proprio=s['proprio'][0],
                    context=s['context'],context_mask=s['context_mask'],
                    num_inference_steps=a.nfe,seed=a.seed+index)['action'].float().cpu()
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if tuple(out.shape)!=(32,7) or not torch.isfinite(out).all():
                raise ValueError('Invalid predicted action shape or non-finite output')
            physical = action_norm.backward(out)
            valid = ~s['action_is_pad'].bool()
            error = (out[valid]-s['action'][valid].float()).abs().mean().item() if valid.any() else None
            rows.append({'dataset_index':index,'prompt':s['prompt'],'cold_start':index==0,
                         'model_call_seconds':seconds,'valid_action_steps':int(valid.sum()),
                         'normalized_open_loop_mae':error,'normalized_action':out.tolist(),
                         'joint_target_rad_plus_gripper_width_m':physical.tolist()})
    report={'schema':'warm.real.transfer-probe.v1','teacher_forced':True,'robot_motion':False,
            'checkpoint':str(a.checkpoint.resolve()),'config_sha256':file_sha256(a.config),
            'normalizer_sha256':file_sha256(Path(str(cfg.data.val.pretrained_norm_stats))),
            'latency_scope':'model call only; excludes image acquisition and decoding; first sample is cold',
            'action_profile':PIPER_JOINT_CONTROL_MODE,'samples':rows}
    with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(a.output.resolve())


if __name__=='__main__':main()
