import hashlib,json
import pytest
from scripts.run_rebuttal_training_segment import validate_parent_archive

def make_plan(tmp_path):
    receipt=tmp_path/'receipt.json'
    receipt.write_text(json.dumps(dict(status='verified_byte_exact_reconstruction',
        role='pooled_memory_prior',optimizer_updates=100,checkpoint_sha256='weights',source_sha256='source')))
    return dict(parent_archive_receipt=str(receipt),parent_archive_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
        role='pooled_memory_prior',start_update=100,parent_checkpoint_sha256='weights',training_source_sha256='source')

def test_real_json_string_receipt_path(tmp_path):
    plan=json.loads(json.dumps(make_plan(tmp_path)))
    validate_parent_archive(plan)

@pytest.mark.parametrize('key,value',[('parent_archive_receipt_sha256','tampered'),('role','full'),
    ('start_update',200),('parent_checkpoint_sha256','other'),('training_source_sha256','different')])
def test_cross_lineage_receipt_rejected(tmp_path,key,value):
    plan=make_plan(tmp_path);plan[key]=value
    with pytest.raises(ValueError,match='independently preserved'):validate_parent_archive(plan)
