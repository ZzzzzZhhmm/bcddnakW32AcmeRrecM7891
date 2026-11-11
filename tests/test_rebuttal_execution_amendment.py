"""Budget changes cannot silently alter the frozen science or erase unknowns."""
import importlib.util
from pathlib import Path
import datetime
import hashlib
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('execution_budget',Path(__file__).resolve().parents[1]/'scripts/run_rebuttal_execution_amendment.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

class ExecutionAmendment(unittest.TestCase):
    def setUp(self):
        self.plan=dict(phase='formal',task='battery_try',first_admitted_utc='2026-11-11T10:14:31+00:00',absolute_deadline_utc='2026-11-11T22:14:31+00:00',controller_source_sha256='frozen',worker_max_seconds=2400)
        self.a=dict(schema='warm.rebuttal.execution-time-amendment.v1',original_plan_sha256='plan',first_admitted_utc=self.plan['first_admitted_utc'],absolute_deadline_utc=self.plan['absolute_deadline_utc'],frozen_controller_sha256='frozen',effective_worker_max_seconds=3600,applies_to_roles=['full','pooled_memory_prior'],scientific_fields_changed=[],maximum_parallel_formal_slots=6,original_campaign_gpu_hour_ceiling=120,reason='Three predeclared BT attempts reached the engineering limit before1000commands')
        self.now=datetime.datetime(2026,11,11,17,tzinfo=datetime.timezone.utc)
    def test_only_worker_limit_changes_without_mutating_plan(self):
        result=M.qualify(self.a,self.plan,'plan',self.now)
        self.assertEqual(result,dict(self.plan,worker_max_seconds=3600));self.assertEqual(self.plan['worker_max_seconds'],2400)
    def test_changed_deadline_role_or_science_rejected(self):
        for change in [dict(absolute_deadline_utc='2026-11-12T22:14:31+00:00'),dict(applies_to_roles=['full']),dict(scientific_fields_changed=['checkpoint']),dict(effective_worker_max_seconds=7200)]:
            with self.assertRaises(ValueError):M.qualify(dict(self.a,**change),self.plan,'plan',self.now)
        with self.assertRaises(ValueError):M.qualify(self.a,self.plan,'wrong',self.now)
        with self.assertRaises(ValueError):M.qualify(self.a,self.plan,'plan',self.now+datetime.timedelta(days=1))
    def test_stale_loading_is_unknown_only_with_matching_terminal_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=Path(tmp)/'attempt/worker';worker.mkdir(parents=True);p=worker/'result.json';p.write_text('{}');ex=worker.parent/'exit.json';ex.write_text('{"exit_code":-15}');row=dict(status='loading_typed_policy',success=None)
            proof=dict(attempts={str(worker.parent.resolve()):dict(actual_exit_code=-15,result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),exit_sha256=hashlib.sha256(ex.read_bytes()).hexdigest())})
            view=M.terminal_view(row,worker,proof);self.assertIsNone(view['success']);self.assertEqual(row['status'],'loading_typed_policy');self.assertTrue(view['status'].startswith('unknown'))
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):M.terminal_view(row,worker,proof)

    def test_RB_terminal_status_resume_keeps_original_worker_budget(self):
        plan=dict(self.plan,task='rearrange_blocks')
        amendment=dict(self.a,schema='warm.rebuttal.terminal-status-amendment.v1',
                       effective_worker_max_seconds=2400,
                       reason='Known externally interrupted RB role with archived actual exit')
        self.assertEqual(M.qualify(amendment,plan,'plan',self.now),plan)
        with self.assertRaises(ValueError):
            M.qualify(dict(amendment,effective_worker_max_seconds=3600),plan,'plan',self.now)

if __name__=='__main__':unittest.main()
