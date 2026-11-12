"""Audit and archive-bound-report fixed PB pools on a zero-GPU ACP worker.

Waits only for the existing GPU supervisor's immutable terminal receipts. It
never launches, retries, changes, or extends any scientific experiment.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import datetime
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastwam.research.pb_pool_audit import audit_pool, read, sha, verify_archive


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def validate_plan(plan):
    if (plan.get('schema') != 'warm.PB-terminal-offline-audit.v1'
            or type(plan.get('GPUs')) is not int or plan['GPUs'] != 0
            or plan.get('queries') != [8, 48, 96] or plan.get('max_cpu_workers') != 3
            or plan.get('scientific_jobs_read_only') is not True
            or plan.get('original_windows_unchanged') is not True
            or plan.get('new_policy_or_physical_actions') != 0
            or [q['query'] for q in plan.get('query_plans', [])] != [8, 48, 96]):
        raise ValueError('Only the three fixed queries on a read-only zero-GPU worker are allowed')
    datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
    output = Path(plan['output']).resolve()
    for item in plan['query_plans']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen physical query plan differs')
        physical = read(item['path'])
        if physical['query_replan'] != item['query'] or Path(physical['output']).resolve() != Path(item['root']).resolve():
            raise ValueError('Physical query identity differs')
        if output.is_relative_to(Path(item['root']).resolve()) or Path(item['root']).resolve().is_relative_to(output):
            raise ValueError('Audit output must be disjoint from scientific evidence')
    for item in plan['inputs']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen audit input differs')
    return plan


def terminal_audit(item, terminal, output):
    # The existing GPU supervisor has already closed and archived this query.
    # Verify the original raw tree, then derive measurements without a model.
    directory = Path(output) / f'q{item["query"]}'
    directory.mkdir()
    archive = verify_archive(item['root'], terminal['archive'])
    operator = read(Path(item['root']) / 'operator.json')
    if (operator['status'] != 'complete_original_PB_fixed_query_pool'
            or terminal['exit_code'] != 0 or terminal.get('bounded_termination', False)):
        report = dict(query=item['query'], status='incomplete_physical_query_retained',
                      archive=archive, physical_operator=operator, primary_metrics=None,
                      interrupted_or_invalid_not_imputed=True, no_scientific_retry=True)
    else:
        report = audit_pool(item['root'], query=item['query'], plan_sha256=item['sha256'])
        report.update(status='verified_fixed_query_pool', archive=archive)
        with (directory / 'centered_effects.csv').open('x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['reference', 'complete_H32_candidates', 'valid_candidates', 'centered_MSE', 'absolute_MSE'])
            for name, value in report['metrics']['references'].items():
                writer.writerow([name, report['metrics']['complete_endpoints'], report['metrics']['valid_candidates'], value['centered_mse'], value['absolute_mse']])
        m = report['metrics'];refs = m['references']
        text = ('% Fixed-state diagnostic; one source episode, no population interval.\n'
                f"Press Button q{item['query']} & {m['complete_endpoints']}/{m['valid_candidates']} & "
                + ' & '.join(f"{refs[n]['centered_mse']:.6f}" for n in ['predicted', 'historical', 'constant_query'])
                + r' \\' + '\n') if refs else '% Fewer than two complete-H32 candidates; no MSE estimate.\n'
        (directory / 'centered_effects_row.tex').write_text(text, encoding='utf-8')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if refs:
                fig, ax = plt.subplots(figsize=(5.6, 3.5))
                names = ['predicted', 'historical', 'constant_query']
                ax.bar(range(3), [refs[n]['centered_mse'] for n in names], color=['#33778c', '#8c8c8c', '#b6a07a'])
                ax.set_xticks(range(3), ['Predicted', 'Historical', 'Constant'])
                ax.set_ylabel('Centered effect MSE')
                ax.set_title(f"Press Button q{item['query']}: identical H32 support ({m['complete_endpoints']}/{m['valid_candidates']})")
                fig.text(.5, .02, 'One fixed state; incomplete endpoints omitted for all references', ha='center', fontsize=8)
                fig.tight_layout(rect=(0, .05, 1, 1))
                fig.savefig(directory / 'centered_effects.pdf')
                fig.savefig(directory / 'centered_effects.png', dpi=160)
                plt.close(fig)
            report['figure_status'] = 'complete' if refs else 'no_identified_estimate'
        except Exception as exc:
            report['figure_status'] = 'failed_preserve_numeric_results'
            report['figure_error'] = repr(exc)
    report.update(audited_utc=now().isoformat(),GPUs=0,model_or_simulator_imported=False,publication_authorized=False)
    write(directory / 'audit.json', report)
    return report


def main():
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if sha(args.plan) != args.expected_plan_sha256:
        raise ValueError('Audit plan identity differs')
    plan = validate_plan(read(args.plan))
    if args.validate_only:
        print(json.dumps(dict(status='CPU_plan_validated',GPUs=0,new_physical_actions=0)))
        return
    output = Path(plan['output']); output.mkdir(exist_ok=True)
    lock = (output / 'audit.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'operator.json').exists():
        raise ValueError('An existing or unknown audit invocation cannot be duplicated')
    end = datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
    state = dict(status='waiting_existing_physical_receipts',started_utc=now().isoformat(),absolute_deadline_utc=end.isoformat(),GPUs=0,queries={},new_physical_actions=0)
    def save():
        state['updated_utc']=now().isoformat();tmp=output/'operator.atomic';tmp.write_text(json.dumps(state,indent=2));tmp.replace(output/'operator.json')
    save();pending={};submitted=set()
    with ProcessPoolExecutor(max_workers=3) as executor:
        while now() < end:
            parent = read(Path(plan['physical_stage_root']) / 'operator.json')
            for item in plan['query_plans']:
                q=item['query'];terminal=parent.get('queries',{}).get(str(q))
                if q not in submitted and terminal and terminal.get('archive'):
                    pending[q]=executor.submit(terminal_audit,item,terminal,str(output));submitted.add(q)
            for q,future in list(pending.items()):
                if future.done():
                    try:r=future.result();state['queries'][str(q)]=dict(status=r['status'],audit_path=str(output/f'q{q}/audit.json'),audit_sha256=sha(output/f'q{q}/audit.json'))
                    except Exception as exc:state['queries'][str(q)]=dict(status='audit_failed_raw_unchanged',error=repr(exc),no_automatic_retry=True)
                    del pending[q]
            terminal_parent=parent['status'] in ['terminal_fixed_PB_stage_all_existing_output_archived','failed_or_interrupted_preserve_all']
            if terminal_parent and not pending:
                for q in plan['queries']:
                    if q not in submitted:state['queries'][str(q)]=dict(status='no_archived_terminal_query',physical_status=parent.get('queries',{}).get(str(q)),not_imputed=True)
                break
            if len(state['queries'])==3 and not pending:break
            save();time.sleep(10)
    state.update(status='terminal_offline_audit_window',ended_utc=now().isoformat(),all_three_verified=all(state['queries'].get(str(q),{}).get('status')=='verified_fixed_query_pool' for q in plan['queries']))
    for q in plan['queries']:
        state['queries'].setdefault(str(q),dict(status='not_audited_before_fixed_CPU_deadline',not_imputed=True))
    save()


if __name__ == '__main__':
    main()
