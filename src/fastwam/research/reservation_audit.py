"""Reconcile unused tails of closed admission reservations, never GPU invoices.

Only immutable timing/terminal records are read. Scientific result files and
policy outcomes are neither inputs nor selection criteria.
"""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path


def bound(ref):
    p=Path(ref['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=ref['sha256']:
        raise ValueError('Bound accounting input changed')
    return json.loads(p.read_text(encoding='utf-8'))


def reconcile(plan,*,now=None):
    now=datetime.now(timezone.utc) if now is None else now
    if plan.get('schema')!='warm.closed-reservation-reconciliation.v1' or type(plan.get('GPUs')) is not int or plan['GPUs']!=0:
        raise ValueError('Only a zero-GPU accounting audit is supported')
    state=bound(plan['state']);before=state['authoritative_active_execution']['reserved_upper_GPU_hours']
    if before!=plan['expected_upper_before'] or not math.isfinite(before) or not 0<before<=120:
        raise ValueError('Accounting baseline changed')
    already=set(state.get('credited_closed_reservation_ids',[]));lineage={};previous=None
    for row in plan['lineage']:
        identity=row['registration']['path'];r=bound(row['registration'])
        first=datetime.fromisoformat(r['stage_first_admitted_utc']);end=datetime.fromisoformat(r['stage_absolute_deadline_utc'])
        hours=(end-first).total_seconds()/3600
        inc=r.get('new_resource_reserved_hours_upper_bound',r.get('new_resource_GPU_hours_upper_bound'))
        post=r.get('cumulative_reserved_upper_GPU_hours',r.get('combined_reserved_GPU_hours_upper_bound'))
        if identity in lineage or inc is None or post is None or not all(math.isfinite(v) for v in (inc,post)) or not 0<hours<=12 or not 0<=post-inc<=post<=120 or abs(inc-hours)>1e-9:
            raise ValueError('Duplicate or inconsistent charged reservation')
        if previous is not None and abs(post-inc-previous)>1e-9:
            raise ValueError('The charged lineage has a gap or was counted twice')
        lineage[identity]=(first,end,hours,row['registration']);previous=post
    if previous is None or abs(previous-before)>1e-9:raise ValueError('Charged lineage does not reach the current baseline')
    live=bound(plan['process_attestation'])
    age=(now-datetime.fromisoformat(live['utc'])).total_seconds()
    if live.get('active_old_stage_paths') or not 0<=age<=300:raise ValueError('Old execution is active or its inspection is stale')
    rows=[];seen=set()
    for row in plan['closed']:
        identity=row['registration']['path']
        if identity not in lineage or identity in seen or identity in already:
            raise ValueError('An uncharged or already credited reservation cannot be released')
        seen.add(identity);first,end,hours,ref=lineage[identity]
        stage=Path(row['stage']).resolve()
        if (row['registration']!=ref or Path(identity).resolve().parent!=stage
            or not all(Path(row[k]['path']).resolve().is_relative_to(stage) for k in ('operator','terminal'))):
            raise ValueError('Terminal records do not belong to the charged stage')
        if now<end:raise ValueError('The original reservation window has not closed')
        op=bound(row['operator']);terminal=bound(row['terminal'])
        if op.get('status') not in ('terminal_fixed_roles_invocation_complete','terminal_fixed_invocation_complete_or_window_bound','failed_or_interrupted_preserve_all'):
            raise ValueError('A nonterminal invocation retains its reservation')
        if row['stage'] not in live['inspected_old_stage_paths']:raise ValueError('The old stage process was not inspected')
        times=[datetime.fromisoformat(terminal.get('end_utc',terminal.get('ended_utc')))]
        for key in ['ended_utc','updated_utc']:
            if op.get(key):times.append(datetime.fromisoformat(op[key]))
        ended=max(times)
        if not first<=ended<=now:raise ValueError('Terminal time is outside its bound invocation')
        if 'exit_code' in terminal:
            if type(terminal['exit_code']) is not int:raise ValueError('Unknown exit cannot release a reservation')
        elif terminal.get('status')!=op['status']:raise ValueError('Direct supervisor terminal receipt differs')
        if op['status']=='failed_or_interrupted_preserve_all':
            ref=row['retained_raw_archive']
            if hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()!=ref['sha256']:
                raise ValueError('Interrupted raw evidence must remain preserved')
        credit=max(0,(end-ended).total_seconds()/3600)
        rows.append(dict(reservation_id=identity,stage=row['stage'],original_first_utc=first.isoformat(),original_deadline_utc=end.isoformat(),verified_terminal_utc=ended.isoformat(),original_reserved_GPU_hours=hours,released_unused_tail_GPU_hours=credit))
    if not rows:raise ValueError('At least one independently closed reservation is required')
    credit=sum(r['released_unused_tail_GPU_hours'] for r in rows);after=before-credit
    if not 0<=after<=before:raise ValueError('Accounting credit exceeds its original reservation')
    return dict(schema='warm.closed-reservation-reconciliation-result.v1',utc=now.isoformat(),upper_before=before,released_unused_tails=credit,upper_after=after,remaining_within120=120-after,credited_reservation_ids=sorted(seen),closed=rows,original_windows_modified=False,scientific_outcome_files_read=False,new_GPU_or_physical_actions=0,actual_billed_GPU_hours_estimated=False,scope='Conservative admission-reservation accounting only;not invoice reconciliation,new GPU quota,or empirical performance evidence.')
