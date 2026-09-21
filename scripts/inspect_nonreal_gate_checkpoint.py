#!/usr/bin/env python3
"""Read only the small gate tensors from a trusted local checkpoint on CPU."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=True)
    gate=payload['warm_retrospection_modules']['source_confidence_gate']
    tensors={name:dict(shape=list(tensor.shape),dtype=str(tensor.dtype),nonzero=int(torch.count_nonzero(tensor)),
                       minimum=float(tensor.min()),maximum=float(tensor.max()),norm=float(tensor.float().norm()))
             for name,tensor in gate.items()}
    report=dict(schema='warm.gate-checkpoint-inspection.v1',checkpoint=str(args.checkpoint),
                checkpoint_version=payload['warm_retrospection']['version'],tensors=tensors,
                final_weight_exactly_zero=bool((gate['network.2.weight']==0).all()),
                final_bias_exactly_minus_two=bool((gate['network.2.bias']==-2).all()),
                limitation='CPU tensor inspection, not a gradient or optimizer update test. No checkpoint modification.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(report))


if __name__=='__main__':
    main()
