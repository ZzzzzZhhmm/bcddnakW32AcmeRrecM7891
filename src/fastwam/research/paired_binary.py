"""Conservative finite-sample bounds for paired binary episode differences.

Treat gain/harm as two binomial marginals of a paired multinomial. Separate
Clopper-Pearson tails and a union bound require no independence between them.
Zero discordances therefore never imply a zero-width uncertainty interval.
"""
from __future__ import annotations

import math


def _binomial_cdf(k, n, p):
    if k < 0:return 0.0
    if k >= n or p == 0:return 1.0
    if p == 1:return 0.0
    logs=[math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
          +j*math.log(p)+(n-j)*math.log1p(-p) for j in range(k+1)]
    peak=max(logs)
    return min(1.0,math.exp(peak)*math.fsum(math.exp(v-peak) for v in logs))


def binomial_tail_bounds(k, n, tail):
    """Lower and upper one-sided exact bounds, each with error at most tail."""
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n < 1:
        raise ValueError('integer binomial count and positive denominator required')
    if not 0 < tail < .5:raise ValueError('tail error must lie in (0,.5)')
    if k == 0:lower=0.0
    else:
        # P_p[X >= k] = tail; use the reflected lower tail to avoid 1-CDF.
        left,right=0.0,1.0
        for _ in range(70):
            middle=(left+right)/2
            if _binomial_cdf(n-k,n,1-middle) < tail:left=middle
            else:right=middle
        lower=(left+right)/2
    if k == n:upper=1.0
    else:
        left,right=0.0,1.0
        for _ in range(70):
            middle=(left+right)/2
            if _binomial_cdf(k,n,middle) > tail:left=middle
            else:right=middle
        upper=(left+right)/2
    return lower,upper


def paired_difference_bounds(gain, harm, n, *, alpha=.05, simultaneous_tasks=1):
    """Equal-task macro combines these task bounds with Bonferroni coverage."""
    if any(type(x) is not int for x in (gain,harm,n,simultaneous_tasks)) or gain<0 or harm<0 or gain+harm>n or n<1 or simultaneous_tasks<1:
        raise ValueError('valid mutually exclusive paired counts required')
    if not 0 < alpha < .5:raise ValueError('alpha must lie in (0,.5)')
    tail=alpha/(4*simultaneous_tasks)
    g_lower,g_upper=binomial_tail_bounds(gain,n,tail)
    h_lower,h_upper=binomial_tail_bounds(harm,n,tail)
    one_tail=alpha/(2*simultaneous_tasks)
    ni_lower=binomial_tail_bounds(gain,n,one_tail)[0]-binomial_tail_bounds(harm,n,one_tail)[1]
    return dict(gain=gain,harm=harm,pairs=n,difference=(gain-harm)/n,
        ci95_conservative=[g_lower-h_upper,g_upper-h_lower],
        one_sided95_lower=ni_lower,alpha=alpha,simultaneous_tasks=simultaneous_tasks,
        method='Clopper-Pearson marginal tails with Bonferroni union bound; no gain/harm independence assumed')


def fixed_task_paired_report(rows, *, tasks, planned_pairs_per_task, margin=.02, alpha=.05):
    """Unknown/invalid pairs remain explicit; incomplete cohorts cannot pass NI."""
    if not tasks or len(tasks)!=len(set(tasks)) or type(planned_pairs_per_task) is not int or planned_pairs_per_task<1 or not 0<margin<1:
        raise ValueError('fixed distinct tasks, planned count and declared margin required')
    buckets={task:[] for task in tasks};seen=set();identities=set()
    for row in rows:
        task=row['task'];key=(task,row['episode_id'])
        if task not in buckets or key in seen:raise ValueError('unexpected task or duplicate episode')
        if row.get('phase')!='formal':raise ValueError('qualification pilots cannot enter formal inference')
        seen.add(key);buckets[task].append(row)
        identities.add(tuple(row.get(k) for k in ('checkpoint_sha256','plan_sha256','search_condition')))
    if len(identities)>1:raise ValueError('do not pool models, protocols or search interventions')
    report=dict(tasks={},fixed_tasks=list(tasks),margin=margin,alpha=alpha,
        planned_pairs_per_task=planned_pairs_per_task,macro_difference=None,macro_ci95=None,
        macro_one_sided95_lower=None,noninferiority_established=False,cohort_complete=False)
    for task,values in buckets.items():
        gain=harm=n=unknown=0
        if len(values)>planned_pairs_per_task:raise ValueError('cohort exceeds prespecified count')
        for row in values:
            a=row.get('roles',{}).get('full',{}).get('success')
            b=row.get('roles',{}).get('candidate_null',{}).get('success')
            if row.get('status')!='complete' or type(a) is not int or type(b) is not int or a not in (0,1) or b not in (0,1):
                unknown+=1;continue
            n+=1;gain+=int(a==1 and b==0);harm+=int(a==0 and b==1)
        scored=paired_difference_bounds(gain,harm,n,alpha=alpha,simultaneous_tasks=len(tasks)) if n else None
        report['tasks'][task]=dict(attempted=len(values),unknown=unknown,not_observed=planned_pairs_per_task-len(values),
            complete_pairs=n,complete_case_bounds=scored)
    complete=all(v['complete_pairs']==planned_pairs_per_task for v in report['tasks'].values())
    report['cohort_complete']=complete
    if all(v['complete_case_bounds'] is not None for v in report['tasks'].values()):
        scores=[v['complete_case_bounds'] for v in report['tasks'].values()]
        report['macro_difference']=sum(s['difference'] for s in scores)/len(tasks)
        report['macro_ci95']=[sum(s['ci95_conservative'][i] for s in scores)/len(tasks) for i in (0,1)]
        report['macro_one_sided95_lower']=sum(s['one_sided95_lower'] for s in scores)/len(tasks)
        report['noninferiority_established']=complete and report['macro_one_sided95_lower'] > -margin
    report['scope']='Fixed-task episode-seed uncertainty only. Complete-case intervals are descriptive while any planned pair is missing/invalid.'
    return report
