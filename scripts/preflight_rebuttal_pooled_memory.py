#!/usr/bin/env python3
"""Check a research pooler on one fixed train sample and a sealed online record.

This is a module/interface preflight, not an optimizer update or policy rollout.
History-gist inputs are explicit zeros here; causal history routing is covered
separately by the model tests. No future teachers are forwarded to the pooler.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from omegaconf import OmegaConf
    from fastwam.runtime import build_datasets
    from fastwam.models.warm.adapter_config import RetrospectiveEventAdapterConfig
    from fastwam.models.warm.retrospection_config import WarmRetrospectionConfig
    from fastwam.models.warm.retrospection_model import _warp_candidate_actions
    from fastwam.research.pooled_memory_prior import PooledMemoryEvent, POOL_ARCHITECTURE
    from fastwam.utils.misc import register_work_dir
    from scripts.nonreal_job import source_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--sealed-context", type=Path, required=True)
    parser.add_argument("--training-index", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.training_index < 0 or not args.output.is_absolute() or args.output.resolve().is_relative_to(ROOT):
        raise ValueError("fixed nonnegative index and absolute output outside source are required")
    if args.output.exists():
        raise ValueError("preserve the previous preflight receipt")
    # The common dataset copies its immutable normalization statistics here.
    register_work_dir(args.output.with_suffix(".work"))
    recipe = OmegaConf.load(args.recipe)
    cfg = WarmRetrospectionConfig.from_dict(OmegaConf.to_container(recipe.model.retrospection, resolve=True))
    adapter_cfg = RetrospectiveEventAdapterConfig(
        action_dim=cfg.action_dim, action_horizon=cfg.action_horizon, timing_dim=cfg.timing_dim,
        world_dim=cfg.semantic_dim, gist_dim=cfg.gist_dim, proprio_dim=cfg.proprio_dim,
        text_dim=cfg.text_dim, event_dim=cfg.semantic_dim, model_dim=cfg.event_model_dim,
        effect_dim=cfg.semantic_dim, num_heads=cfg.event_heads, rho=cfg.residual_rho,
    )
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    torch.manual_seed(3407)
    adapter = PooledMemoryEvent(adapter_cfg, context_dim=cfg.context_dim).to(device=device, dtype=dtype)
    source = source_identity(ROOT)["sha256"]
    dataset, _ = build_datasets(recipe.data, build_validation=False)
    sample = dataset[args.training_index]
    training = {name: value[None] for name,value in sample.items() if torch.is_tensor(value)}
    sealed = torch.load(args.sealed_context, map_location="cpu", weights_only=True)
    forbidden = {"target_effect", "target_action", "current_semantic_teacher", "future_valid_mask", "target_action_valid_mask"}
    if forbidden.intersection(sealed):
        raise ValueError("future teacher present in sealed inference record")
    aliases = dict(candidate_actions="warm_candidate_mu", candidate_valid_mask="warm_candidate_mask",
                   candidate_effect_pre="warm_candidate_effect_pre", candidate_effect_delta="warm_candidate_effect_delta",
                   candidate_context="warm_candidate_context", candidate_timing="warm_candidate_timing",
                   candidate_start_proprio="warm_candidate_start_proprio", query_context="warm_current_context",
                   candidate_normalized_phase="warm_candidate_normalized_phase", candidate_event_ordinal="warm_candidate_event_ordinal")
    train_context = {name: training[key] for name,key in aliases.items()}
    rows = []
    for label, context, proprio in (("fixed_train_sample", train_context, training["proprio"][:,0]),
                                     ("sealed_online_record", sealed, torch.zeros(1,cfg.proprio_dim))):
        def cast(value):
            return value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        context = {key:cast(value) for key,value in context.items() if torch.is_tensor(value)}
        valid = context["candidate_valid_mask"]
        b, k = valid.shape
        proprio = cast(proprio)
        actions = _warp_candidate_actions(context["candidate_actions"], context["candidate_start_proprio"], proprio, valid, cfg)
        phase = context["candidate_normalized_phase"]
        ordinal = context["candidate_event_ordinal"].to(dtype=dtype)
        phase_features = torch.stack((phase,torch.sin(torch.pi*phase),torch.cos(torch.pi*phase),ordinal/(ordinal+8)),dim=-1)
        kwargs = dict(warped_actions=actions, candidate_valid_mask=valid,
                      event_pre_tokens=context["candidate_effect_pre"], event_delta_tokens=context["candidate_effect_delta"],
                      event_delta_mask=valid[...,None].expand(b,k,4), candidate_context=context["candidate_context"],
                      candidate_phase_features=phase_features, gripper_timing=context["candidate_timing"],
                      query_context=context["query_context"], proprio=proprio,
                      gist_tokens=torch.zeros(b,8,cfg.gist_dim,device=device,dtype=dtype))
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        result = adapter(**kwargs)
        if device.type == "cuda": torch.cuda.synchronize()
        forward_s = time.perf_counter()-start
        assert torch.isfinite(result.action_context_tokens).all()
        (result.action_context_tokens.float()*torch.randn_like(result.action_context_tokens).float()).sum().backward()
        gradients = [p.grad for p in adapter.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert bool(adapter.record_projection[0].weight.grad.abs().sum()>0)
        adapter.zero_grad(set_to_none=True)
        rows.append(dict(record=label,batch=b,candidates=k,valid_candidates=int(valid.sum()),
                         output_shape=list(result.action_context_tokens.shape),forward_s=forward_s,
                         all_parameter_gradients_finite=True,history_gist_input="explicit_zero_interface_placeholder",
                         current_proprio_input="actual_training_proprio" if label=="fixed_train_sample" else "explicit_zero_interface_placeholder"))
    assert source_identity(ROOT)["sha256"] == source
    result = dict(status="qualified_module_interface_only",architecture=POOL_ARCHITECTURE,
                  source_sha256=source,adapter_config=asdict(adapter_cfg),context_dim=cfg.context_dim,
                  pool_parameters=sum(p.numel() for p in adapter.parameters()),device=str(device),dtype=str(dtype),
                  fixed_training_index=args.training_index,records=rows,optimizer_updates=0,policy_actions=0,
                  recipe_sha256=hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
                  sealed_context_sha256=hashlib.sha256(args.sealed_context.read_bytes()).hexdigest(),
                  scope="Actual retrieved train/online records, module numerical/interface check only. No full sampler qualification or efficacy result.")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
