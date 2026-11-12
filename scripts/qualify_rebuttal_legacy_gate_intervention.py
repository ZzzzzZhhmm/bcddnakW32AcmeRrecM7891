"""Qualify the intervention against actual frozen v3 code on CPU fixtures.

This is an engineering check, not checkpoint, GPU, or physical qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-source', required=True, type=Path)
    parser.add_argument('--intervention', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.original_source.resolve(strict=True)
    sys.path[:0] = [str(source / 'src'), str(source)]
    import torch
    from tests import test_warm_retrospection_model_torch as original

    spec = importlib.util.spec_from_file_location('qualified_legacy_intervention', args.intervention)
    intervention = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(intervention)
    withdraw = intervention.legacy_candidate_withdrawal
    checks = []

    def checked(name, condition=True):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    def rejects(name, thunk, message):
        try:
            thunk()
        except (ValueError, RuntimeError) as error:
            checked(name, message in str(error))
        else:
            raise AssertionError(name + ' did not reject')

    torch.manual_seed(3407)
    model = original._model().eval()
    # The tiny fixture initializes this projection to zero. Enable an arbitrary
    # fixed nonzero projection before all checks so retained history is observable.
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(model.gist_to_text.weight)
    context = original._source_context(with_teachers=False)
    rng = torch.get_rng_state().clone()

    def resolve(ctx=context):
        torch.set_rng_state(rng.clone())
        return original._resolve(model, ctx, phase='infer')[1]

    def null(ctx=context):
        with withdraw(model) as witness:
            output = resolve(ctx)
        return output, witness

    def enter_only():
        with withdraw(model):
            pass

    def clean():
        return ('_resolve_action_source' not in model.__dict__ and
                not any(m._forward_hooks or m._forward_pre_hooks for m in
                        [model.source_confidence_gate, model.retrospective_gist,
                         model.gist_to_text, model.action_context_to_text]))

    rejects('gradient-enabled call rejected', enter_only, 'inference_mode')
    with torch.no_grad():
        model.train()
        rejects('training model rejected', enter_only, 'eval-only')
        model.eval()
        hook = model.source_confidence_gate.register_forward_hook(lambda *x: None)
        rejects('preexisting hook rejected', enter_only, 'preexisting hooks')
        checked('foreign hook retained', hook.id in model.source_confidence_gate._forward_hooks)
        hook.remove()
        full_before = resolve()
        null_base, witness = null()
        checked('exact Gaussian initialization', torch.equal(null_base.source, null_base.base_gaussian))
        checked('both candidate paths withdrawn', witness['conditioning_exactly_query_gist_and_zero_candidate_tokens'])
        checked('natural gate preserved in witness', witness['natural_probability'][0].item() > 0 and
                witness['natural_probability'][1].item() == 0)
        checked('retrieval metadata retained', torch.equal(full_before.memory_mask, null_base.memory_mask))
        checked('not certified as natural withdrawal', not witness['natural_withdrawal_certified'])
        varied = replace(context,
            candidate_actions=context.candidate_actions * 1.7,
            candidate_effect_pre=context.candidate_effect_pre * 1.2,
            candidate_effect_delta=context.candidate_effect_delta * 2,
            candidate_timing=context.candidate_timing * .8)
        null_varied, _ = null(varied)
        checked('candidate perturbation leaves withdrawn source unchanged', torch.equal(null_base.source, null_varied.source))
        checked('candidate perturbation leaves withdrawn conditioning unchanged', torch.equal(null_base.conditioning_tokens, null_varied.conditioning_tokens))
        empty = replace(context, **{
            key: torch.zeros_like(getattr(context, key)) for key in [
                'candidate_actions', 'candidate_effect_pre', 'candidate_effect_delta',
                'candidate_timing', 'candidate_context', 'candidate_start_proprio',
                'candidate_support', 'candidate_valid_mask']},
            forced_candidate_indices=torch.full_like(context.forced_candidate_indices, -1))
        null_empty, _ = null(empty)
        checked('empty-bank conditioning agrees with coordinated withdrawal', torch.equal(null_base.conditioning_tokens, null_empty.conditioning_tokens))
        no_history = replace(context, **{
            key: torch.zeros_like(getattr(context, key)) for key in [
                'episode_tokens', 'episode_mask', 'episode_action_summaries', 'episode_action_mask']})
        null_no_history, _ = null(no_history)
        checked('factual episode history still affects conditioning', not torch.equal(null_base.conditioning_tokens, null_no_history.conditioning_tokens))
        full_after = resolve()
        checked('original Full source restored exactly', torch.equal(full_before.source, full_after.source))
        checked('original Full conditioning restored exactly', torch.equal(full_before.conditioning_tokens, full_after.conditioning_tokens))
        checked('all intervention hooks and override removed', clean())
        try:
            with withdraw(model):
                raise LookupError('fixed original exception')
        except LookupError as error:
            checked('original exception retained', str(error) == 'fixed original exception')
        checked('cleanup after exception', clean())
        rejects('incomplete invocation rejected', enter_only, 'not completed')
        def two_calls():
            with withdraw(model):
                resolve()
                resolve()
        rejects('second inference rejected', two_calls, 'one fixed inference')
        checked('cleanup after rejected second inference', clean())
        def wrong_phase():
            with withdraw(model):
                original._resolve(model, context, phase='train')
        rejects('training phase rejected inside scope', wrong_phase, 'only inference')
        checked('cleanup after rejected phase', clean())
    receipt = dict(
        schema='warm.original-v3-withdrawal-CPU-qualification.v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source), source_module_sha256=intervention.LEGACY_MODEL_SHA256,
        intervention_sha256=hashlib.sha256(args.intervention.read_bytes()).hexdigest(),
        qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checks=checks, passed=len(checks), device='cpu', fixture_only=True,
        actual_checkpoint_loaded=False, GPU_replay_qualified=False,
        physical_branches=0, natural_withdrawal_certified=False,
        efficacy_established=False,
    )
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
