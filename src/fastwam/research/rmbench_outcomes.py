"""Independent, declared physical-interaction outcomes for two pinned tasks.

Readers never call check_success, which mutates the official tasks. Local
interaction completion is distinct from final episode success and from the
historical paper's independently annotated applicability cohort.
"""
from __future__ import annotations
import numpy as np

RUBRIC='warm.rmbench.physical-next-interaction.v1'


def _entity_ids(actor):
    if hasattr(actor,'get_links'):return {link.entity.get_global_id() for link in actor.get_links()}
    value=actor.entity if hasattr(actor,'entity') else actor
    return {value.get_global_id()}


def _contact(env,actor):
    robot=_entity_ids(env.robot.left_entity)|_entity_ids(env.robot.right_entity)
    targets=_entity_ids(actor.actor if hasattr(actor,'actor') else actor)
    for contact in env.scene.physx_system.get_contacts():
        a,b=[body.entity.get_global_id() for body in contact.bodies]
        if ((a in robot and b in targets) or (b in robot and a in targets)) and any(np.linalg.norm(p.impulse)>0 for p in contact.points):
            return True
    return False


def observe_task_progress(env):
    """Privileged outcome evidence only; never provided to the policy."""
    row=dict(rubric=RUBRIC,task=env.task_name,terminal_success=bool(env.eval_success),
             take_action_cnt=int(env.take_action_cnt),step_limit=int(env.step_lim))
    if env.task_name=='rearrange_blocks':
        a=np.asarray(env.block1.get_pose().p);b=np.asarray(env.block2.get_pose().p)
        final=np.asarray(env.final_occupy_mat.get_pose().p);mid=np.asarray(env.mid_mat.get_pose().p)
        first=float(np.max(np.abs(a[:2]-final[:2])))
        second=float(np.max(np.abs(b[:2]-mid[:2])))
        center=float(np.max(np.abs(b[:2]-np.array([.13,env.block_y_lim]))))
        row.update(stage_id=int(env.stage_id),press_cnt=int(env.press_cnt),positions=[a.tolist(),b.tolist()],
            first_placement=bool(first<.03 and a[2]<.77 and second<.03 and b[2]<.77),
            final_second_placement=bool(center<.03 and env.is_right_gripper_open()),
            lifted=[bool(a[2]>.781 and _contact(env,env.block1)),bool(b[2]>.781 and _contact(env,env.block2))],
            predicate_margins=[.03-first,.77-float(a[2]),.03-second,.77-float(b[2]),.03-center])
    elif env.task_name=='battery_try':
        poses=[env.battery1.get_pose(),env.battery2.get_pose()]
        targets=[env.target_pose1_p,env.target_pose2_p]
        distances=[float(np.linalg.norm(np.asarray(p.p)[:2]-np.asarray(t)[:2])) for p,t in zip(poses,targets,strict=True)]
        row.update(positions=[p.p.tolist() for p in poses],quaternions=[p.q.tolist() for p in poses],
            in_slots=[d<.02 for d in distances],combination=[int(x) for x in env.get_curr_combination()],
            correct_combination=[int(x) for x in env.correct_combination],dashboard_on=bool(env.check_dashboard_on()),
            lifted=[bool(p.p[2]>.781 and _contact(env,actor)) for p,actor in zip(poses,[env.battery1,env.battery2],strict=True)],
            predicate_margins=[.02-d for d in distances])
    else:raise ValueError('unsupported independent task outcome')
    row['object_fell_below_table']=any(p[2]<.70 for p in row['positions'])
    return row


def label_next_interaction(parent,endpoint,*,valid_branch,seen_combinations=()):
    """Blind to proposal role, ranking and model predictions; retain ambiguity."""
    if parent.get('rubric')!=RUBRIC or endpoint.get('rubric')!=RUBRIC or parent['task']!=endpoint['task']:
        raise ValueError('independent task rubric mismatch')
    result=dict(applicable=None,label_provenance=RUBRIC,scope='Physical next-interaction completion, not generic action quality or final SR',
                terminal_success=bool(endpoint['terminal_success']))
    if not valid_branch:return {**result,'reason':'invalid runtime/branch'}
    if endpoint['terminal_success']:return {**result,'applicable':1,'reason':'official executor success'}
    if endpoint['object_fell_below_table']:return {**result,'applicable':0,'reason':'object fell below table'}
    if endpoint['task']=='rearrange_blocks':
        if endpoint['press_cnt']>1:return {**result,'applicable':0,'reason':'official over-press failure condition'}
        def rank(s):
            if s['stage_id']>=2:return 5
            if s['stage_id']==1:return 4 if s['lifted'][1] else 3
            if s['first_placement']:return 2
            return 1 if s['lifted'][0] else 0
        before,after=rank(parent),rank(endpoint)
        if after>before:return {**result,'applicable':1,'reason':'new ordered physical/task milestone','milestones':[before,after]}
    else:
        before=[2 if s else int(l) for s,l in zip(parent['in_slots'],parent['lifted'],strict=True)]
        after=[2 if s else int(l) for s,l in zip(endpoint['in_slots'],endpoint['lifted'],strict=True)]
        if any(a<b for a,b in zip(after,before,strict=True)):
            return {**result,'applicable':0,'reason':'previous physical milestone regressed','milestones':[before,after]}
        if any(a>b for a,b in zip(after,before,strict=True)):
            return {**result,'applicable':1,'reason':'new grasp/lift or slot placement','milestones':[before,after]}
        combination=tuple(endpoint['combination'])
        if all(endpoint['in_slots']) and combination!=tuple(parent['combination']) and combination not in {tuple(x) for x in seen_combinations}:
            return {**result,'applicable':1,'reason':'previously untested physical polarity configuration reached'}
    if any(abs(float(m))<=.002 for m in parent['predicate_margins']+endpoint['predicate_margins']):
        return {**result,'reason':'within2mm of a geometric predicate boundary; retain for blinded review'}
    return {**result,'applicable':0,'reason':'no completed new physical/task interaction'}
