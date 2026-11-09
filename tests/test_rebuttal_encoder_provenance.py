import unittest
from pathlib import Path
import numpy as np

from fastwam.research.encoder_provenance import runtime_differences,require_feature_parity
from scripts.run_rebuttal_resumable_campaign import role_wrapper


class EncoderProvenanceTests(unittest.TestCase):
    def test_only_platform_provenance_can_differ(self):
        expected=dict(platform='original-kernel',torch='pinned',cuda_device_name='H100',flag=False)
        actual=dict(expected,platform='another-kernel')
        self.assertEqual(set(runtime_differences(expected,actual)),{'platform'})
        self.assertEqual(expected['platform'],'original-kernel')
        for key in ('torch','cuda_device_name','flag'):
            with self.assertRaises(ValueError):runtime_differences(expected,dict(actual,**{key:'changed'}))
        with self.assertRaises(ValueError):runtime_differences(expected,dict(actual,unknown=1))

    def test_all_factual_feature_values_and_dtype_must_match(self):
        expected=np.ones((4,768),dtype=np.float32)
        require_feature_parity(expected,expected.copy())
        changed=expected.copy();changed[-1,-1]=np.nextafter(changed[-1,-1],np.float32(2))
        for bad in (changed,expected.astype(np.float64),expected[:3],np.full((4,768),np.nan,dtype=np.float32)):
            with self.assertRaises(ValueError):require_feature_parity(expected,bad)
        zeros=np.zeros((4,768),dtype=np.float32)
        with self.assertRaises(ValueError):require_feature_parity(zeros,-zeros)

    def test_old_worker_stays_unchanged_and_new_guard_precedes_policy(self):
        plan=dict(core_code='/frozen-core',memory_fraction=.65)
        original=role_wrapper(plan)
        self.assertIn('torch.cuda.set_per_process_memory_fraction(0.65)',original)
        self.assertNotIn('module.install',original)
        guarded=role_wrapper(plan,qualification_plan='/qualification.json',qualification_sha='fixed-sha',receipt='/attempt/proof.json')
        self.assertLess(guarded.index('module.install'),guarded.index('runpy.run_path'))
        self.assertIn(repr(str(Path('/frozen-core')/'scripts/run_rebuttal_rmbench_pairs.py')),guarded)
        compile(guarded,'guarded-role','exec')


if __name__=='__main__':unittest.main()
