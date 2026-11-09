"""Recovery and resumption admit the same durable trial layouts."""
import json
import pytest
from fastwam.research.resumable_pairs import read_committed_trials


@pytest.mark.parametrize('filename',['pairs.jsonl','role_trials.jsonl'])
def test_both_collectors_keep_committed_rows_and_an_unknown_tail(tmp_path,filename):
    row={'episode_id':0,'seed':17,'roles':{'full':{'status':'complete','success':0}}}
    path=tmp_path/filename
    original=json.dumps(row)+'\n'+json.dumps({'episode_id':1})[:-1]
    path.write_text(original)
    assert read_committed_trials(tmp_path)==[row]
    assert path.read_text()==original


def test_role_trials_take_precedence_and_missing_layout_has_no_admitted_rows(tmp_path):
    assert read_committed_trials(tmp_path)==[]
    (tmp_path/'pairs.jsonl').write_text('{"episode_id":1}\n')
    (tmp_path/'role_trials.jsonl').write_text('{"episode_id":0}\n')
    assert read_committed_trials(tmp_path)==[{'episode_id':0}]
