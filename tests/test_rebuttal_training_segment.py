import math
import unittest

from scripts.run_rebuttal_training_segment import segment_config, validate_metrics


class SegmentContractTests(unittest.TestCase):
    def recipe(self):
        return dict(output_dir='old', resume='old-state', run_steps=100, save_every=100,
                    max_steps=30000, batch_size=8, gradient_accumulation_steps=16,
                    seed=3407, learning_rate=5e-5, eval_every=500, model={'unchanged': True})

    def test_original_recipe_and_parent_are_preserved(self):
        before = self.recipe()
        after = segment_config(before, output='new', resume='state200', start=200, end=500)
        self.assertEqual(after['run_steps'], 300)
        self.assertEqual(after['save_every'], 500)
        self.assertEqual(after['eval_every'], 500)
        self.assertEqual(before['save_every'], 100)
        self.assertEqual(after['model'], before['model'])
        self.assertEqual([s for s in range(201, 501) if s % after['save_every'] == 0], [500])

    def test_changed_seed_batch_or_scheduler_is_rejected(self):
        for key in ('max_steps', 'batch_size', 'gradient_accumulation_steps', 'seed'):
            recipe = self.recipe()
            recipe[key] += 1
            with self.assertRaises(ValueError):
                segment_config(recipe, output='new', resume='parent', start=200, end=500)

    def test_no_regression_or_selection_past_fixed_target(self):
        for start, end in ((200, 200), (200, 100), (200, 3001), (0, 500)):
            with self.assertRaises(ValueError):
                segment_config(self.recipe(), output='new', resume='parent', start=start, end=end)

    def test_metrics_need_all_finite_unique_ordered_updates(self):
        rows = [dict(step=s, loss=.2, grad_norm=.1) for s in range(201, 204)]
        validate_metrics(rows, start=200, end=203)
        for bad in (rows[:-1], rows + rows[-1:], list(reversed(rows))):
            with self.assertRaises(ValueError):
                validate_metrics(bad, start=200, end=203)
        for field in ('loss', 'grad_norm'):
            bad = [dict(r) for r in rows]
            bad[0][field] = math.nan
            with self.assertRaises(ValueError):
                validate_metrics(bad, start=200, end=203)


if __name__ == '__main__':
    unittest.main()
