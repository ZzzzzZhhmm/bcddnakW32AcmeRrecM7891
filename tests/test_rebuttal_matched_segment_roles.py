import copy
import unittest
from scripts.run_rebuttal_training_segment import validate_role_recipe

class MatchedRoleTests(unittest.TestCase):
    def recipes(self):
        full={'model':{'_target_':'fastwam.runtime.create_warm_retrospection'}}
        pool={'model':{'_target_':'fastwam.research.pooled_memory_prior.create_pooled_memory_prior',
                       'retrospection':{k:0.0 for k in ('loss_retrieval','loss_effect','loss_gate','loss_adaptation')}}}
        return full,pool
    def test_declared_constructors_remain_distinct(self):
        full,pool=self.recipes();validate_role_recipe('full',full);validate_role_recipe('pooled_memory_prior',pool)
    def test_cross_role_and_undeclared_model_are_rejected(self):
        full,pool=self.recipes()
        for role,cfg in [('full',pool),('pooled_memory_prior',full),('conditioned_prior',pool)]:
            with self.assertRaises(ValueError):validate_role_recipe(role,cfg)
    def test_control_cannot_keep_a_removed_mechanism_loss(self):
        _,pool=self.recipes()
        for key in pool['model']['retrospection']:
            bad=copy.deepcopy(pool);bad['model']['retrospection'][key]=0.05
            with self.assertRaises(ValueError):validate_role_recipe('pooled_memory_prior',bad)

if __name__=='__main__':unittest.main()
