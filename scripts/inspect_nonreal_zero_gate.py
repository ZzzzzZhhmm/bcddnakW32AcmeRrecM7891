#!/usr/bin/env python3
"""Inspect saved ZeRO-1/2 FP32 gate slices without restoring the full model.

Trusted project checkpoints only. Uses the partition concatenation order in
DeepSpeed's bundled zero_to_fp32.py, reading only small gate slices via mmap.
"""
import argparse
import json
import math
from pathlib import Path


def slice_partitions(partitions, start, count):
    import torch
    result=[]
    cursor=0
    for tensor in partitions:
        lo=max(start-cursor,0)
        hi=min(start+count-cursor,tensor.numel())
        if hi>lo:
            result.append(tensor[lo:hi].clone())
        cursor+=tensor.numel()
    if sum(t.numel() for t in result)!=count:
        raise ValueError('partition slice exceeds saved state')
    return torch.cat(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    root=args.state/'pytorch_model'
    model=torch.load(root/'mp_rank_00_model_states.pt',map_location='cpu',mmap=True,weights_only=False)
    world=int(model['dp_world_size'])
    optim=[]
    for rank in range(world):
        path=root/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        value=torch.load(path,map_location='cpu',mmap=True,weights_only=False)['optimizer_state_dict']
        if int(value['zero_stage']) not in (1,2) or max(value['partition_count'])!=world:
            raise ValueError('unsupported or inconsistent ZeRO partitioning')
        optim.append(value)
    rows=[]
    for group,shapes in enumerate(model['param_shapes']):
        partitions=[state['single_partition_of_fp32_groups'][group] for state in optim]
        offset=0
        for name,shape in shapes.items():
            size=math.prod(shape)
            if name.startswith('source_confidence_gate.'):
                value=slice_partitions(partitions,offset,size).view(shape)
                saved=model['module'][name]
                if not torch.equal(value.to(saved.dtype),saved):
                    raise ValueError('reconstructed master does not match saved model after cast')
                row=dict(name=name,shape=list(shape),master_dtype=str(value.dtype),model_dtype=str(saved.dtype),
                         master_min=float(value.min()),master_max=float(value.max()),
                         max_master_minus_model=float((value-saved.float()).abs().max()),
                         master_nonzero=int(torch.count_nonzero(value)),cast_matches_model=True)
                if name.endswith('network.2.bias'):
                    row['master_values']=value.reshape(-1).tolist()
                    row['model_values']=saved.float().reshape(-1).tolist()
                rows.append(row)
            offset+=size
        alignment=2*world
        if alignment*math.ceil(offset/alignment)!=alignment*math.ceil(sum(v.numel() for v in partitions)/alignment):
            raise ValueError('parameter shapes disagree with partition sizes')
    if len(rows)!=4:
        raise ValueError('expected all four gate tensors')
    report=dict(status='gate_master_slices_verified',global_steps=int(model['global_steps']),world_size=world,
                tensors=rows,scope='Read-only saved FP32 gate state; not live resume or full optimizer integrity validation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(report))


if __name__=='__main__':
    main()
