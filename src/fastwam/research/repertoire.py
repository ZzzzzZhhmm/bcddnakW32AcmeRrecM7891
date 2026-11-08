"""Declared search-universe interventions over an unchanged immutable bank.

This never relabels bank provenance or substitutes an unrelated donor payload.
Task-family holdout is a manipulation of available experience; actual relevant
skill coverage must still be independently measured at the sampled queries.
"""
from __future__ import annotations

import hashlib
from types import MethodType
import numpy as np


SKILL_FAMILIES = {
    'rearrange_blocks':('rearrange_blocks','put_back_block','swap_blocks','swap_T','blocks_ranking_try','cover_blocks'),
    'battery_try':('battery_try',),
}


def matched_search_rows(task_indices, task_vocabulary, target_task, *, seed=3407):
    task_indices=np.asarray(task_indices)
    vocabulary=tuple(task_vocabulary)
    if task_indices.ndim!=1 or task_indices.dtype.kind not in 'iu' or not len(task_indices):
        raise ValueError('nonempty integer task-index vector required')
    if len(set(vocabulary))!=len(vocabulary) or target_task not in SKILL_FAMILIES or target_task not in vocabulary:
        raise ValueError('explicit supported task vocabulary/target required')
    if np.any(task_indices<0) or np.any(task_indices>=len(vocabulary)):
        raise ValueError('bank task index lies outside the declared vocabulary')
    excluded_names=SKILL_FAMILIES[target_task]
    excluded_ids=[vocabulary.index(name) for name in excluded_names if name in vocabulary]
    heldout=np.flatnonzero(~np.isin(task_indices,excluded_ids)).astype(np.int64)
    target=np.flatnonzero(task_indices==vocabulary.index(target_task))
    if not len(target) or not len(heldout):raise ValueError('both target experience and nonempty heldout background required')
    count=len(heldout);rng=np.random.default_rng(seed)
    if len(target)>=count:
        # Maximal clean-target representation, fixed without model outcomes.
        clean=np.sort(rng.choice(target,size=count,replace=False))
    else:
        background=np.flatnonzero(task_indices!=vocabulary.index(target_task))
        clean=np.sort(np.concatenate((target,rng.choice(background,size=count-len(target),replace=False))))
    result={'clean_size_matched':np.ascontiguousarray(clean,dtype=np.int64),
            'target_skill_family_holdout':np.ascontiguousarray(heldout,dtype=np.int64)}
    for rows in result.values():rows.flags.writeable=False
    return result,dict(target_task=target_task,excluded_task_families=list(excluded_names),seed=seed,
        original_rows=len(task_indices),rows_per_condition=count,
        actual_absent_skill_claim='Family-held-out availability only; zero relevant retrieved coverage requires independent execution labels.',
        counts={condition:{name:int(np.sum(task_indices[rows]==i)) for i,name in enumerate(vocabulary)}
                for condition,rows in result.items()})


def restricted_cosine_search(keys, query, rows, *, top_k, continuation_rows=()):
    """Exact cosine, stable global-row ties, same allowed continuation priority."""
    keys=np.asarray(keys);query=np.asarray(query,dtype=np.float64);rows=np.asarray(rows)
    if keys.ndim!=2 or query.shape!=(keys.shape[1],) or rows.ndim!=1 or rows.dtype.kind not in 'iu':
        raise ValueError('incompatible keys/query/eligible rows')
    if type(top_k) is not int or top_k<1 or len(rows)<top_k:
        raise ValueError('eligible universe must contain at least K distinct events')
    if np.any(rows<0) or np.any(rows>=len(keys)) or len(np.unique(rows))!=len(rows):
        raise ValueError('eligible bank rows must be unique and in range')
    if not np.isfinite(query).all() or np.linalg.norm(query)<=0:
        raise ValueError('finite nonzero query required')
    selected_keys=np.asarray(keys[rows],dtype=np.float64)
    norms=np.linalg.norm(selected_keys,axis=1)
    if not np.isfinite(selected_keys).all() or np.any(norms<=0):raise ValueError('finite nonzero bank keys required')
    scores=np.clip(selected_keys@query/(norms*np.linalg.norm(query)),-1,1)
    order=np.lexsort((rows,-scores))
    top=rows[order[:top_k]]
    allowed=set(rows.tolist());reserved=[]
    for value in continuation_rows:
        value=int(value)
        if value<0 or value>=len(keys):raise ValueError('continuation row outside bank')
        if value in allowed and value not in reserved:reserved.append(value)
    selected=reserved[:top_k]+[int(r) for r in top if int(r) not in reserved]
    selected=selected[:top_k]
    score_map={int(row):float(score) for row,score in zip(rows,scores,strict=True)}
    selected.sort(key=lambda row:(-score_map[row],row))
    return np.ascontiguousarray(selected,dtype=np.int64),np.asarray([score_map[row] for row in selected],dtype=np.float64)


def install_search_restriction(retriever, eligible_rows):
    """Install once before episode admission; retain and report original bank IDs."""
    if retriever._active_episode_index is not None or '_search_full_bank' in vars(retriever):
        raise ValueError('install a declared search intervention once, before any episode')
    rows=np.array(eligible_rows,dtype=np.int64,copy=True)
    if len(rows)<retriever._online_run_contract.top_k:raise ValueError('holdout bank too small for deployed K')
    if np.any(rows<0) or np.any(rows>=len(retriever._bank)) or len(np.unique(rows))!=len(rows):
        raise ValueError('invalid eligible row identity')
    rows.flags.writeable=False
    def search(self,context_key):
        return restricted_cosine_search(self._bank.context_keys,context_key,rows,
            top_k=self._online_run_contract.top_k,continuation_rows=self._continuation_rows)
    retriever._search_full_bank=MethodType(search,retriever)
    return dict(kind='declared_search_universe_intervention',eligible_rows=len(rows),
        eligible_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        original_bank_content_sha256=retriever.online_run_contract.bank_content_sha256,
        scope='Original bank bytes and event IDs retained; search universe restricted, not the original full-bank protocol.')
