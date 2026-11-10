import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'warm_runtime_observer', ROOT / 'scripts/run_rebuttal_training_with_runtime_trace.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RuntimeObserverTests(unittest.TestCase):
    def test_preserves_exact_returned_object_and_call_arguments(self):
        result = {'cuda_matmul_allow_tf32': False}
        seen = []
        def capture(*args, **kwargs):
            seen.append((args, kwargs))
            return result
        with tempfile.TemporaryDirectory() as tmp:
            observer = module.RuntimeObserver(capture, dict(result), Path(tmp) / 'trace')
            token = object()
            self.assertIs(observer(token, rank=0), result)
            self.assertEqual(seen, [((token,), {'rank': 0})])
            saved = json.loads((observer.output / 'runtime-0001.json').read_text())
            self.assertEqual(saved['differences'], [])
            self.assertFalse(saved['formal_resume_validation_bypassed'])
            self.assertFalse(saved['resume_admission_inferred'])

    def test_keeps_deepspeed_precision_and_missing_field_differences(self):
        expected = {'deepspeed_config': {'bf16': {'enabled': True}}, 'tf32': False, 'parent_only': 1}
        observed = {'deepspeed_config': {'bf16': {'enabled': False}}, 'tf32': True, 'new_field': 0}
        differences = module.differences(expected, observed)
        self.assertEqual({r['field'] for r in differences},
                         {'deepspeed_config.bf16.enabled', 'tf32', 'parent_only', 'new_field'})
        self.assertFalse(next(r for r in differences if r['field'] == 'parent_only')['observed_present'])
        self.assertFalse(next(r for r in differences if r['field'] == 'new_field')['expected_present'])
        self.assertEqual(module.differences({'flag': False}, {'flag': 0})[0]['field'], 'flag')

    def test_original_exception_propagates_without_a_fabricated_runtime(self):
        failure = RuntimeError('original capture failed')
        def capture():
            raise failure
        with tempfile.TemporaryDirectory() as tmp:
            observer = module.RuntimeObserver(capture, {}, Path(tmp) / 'trace')
            with self.assertRaises(RuntimeError) as raised:
                observer()
            self.assertIs(raised.exception, failure)
            self.assertEqual(observer.calls, 0)
            self.assertEqual(list(observer.output.iterdir()), [])

    def test_existing_trace_directory_is_not_reused_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'trace';output.mkdir()
            (output / 'original').write_text('retained')
            with self.assertRaises(FileExistsError):
                module.RuntimeObserver(lambda: {}, {}, output)
            self.assertEqual((output / 'original').read_text(), 'retained')

    def test_retains_each_capture_even_when_runtime_changes(self):
        values = [{'deterministic': False}, {'deterministic': True}]
        with tempfile.TemporaryDirectory() as tmp:
            observer = module.RuntimeObserver(lambda: values.pop(0), {'deterministic': False}, Path(tmp) / 'trace')
            first = observer();second = observer()
            self.assertFalse(first['deterministic']);self.assertTrue(second['deterministic'])
            records = [json.loads((observer.output / f'runtime-{i:04d}.json').read_text()) for i in (1, 2)]
            self.assertEqual(records[0]['differences'], [])
            self.assertEqual(records[1]['differences'][0]['field'], 'deterministic')
            self.assertEqual(records[0]['capture_call'], 1)
            self.assertEqual(records[1]['capture_call'], 2)


    def test_reads_serialized_attestation_runtime_without_json_alias(self):
        runtime = {'cuda_matmul_allow_tf32': False}
        parent = dict(schema='warm.training-attestation', actual_global_step=200,
                      source_policy='fixed_context_top1', training_runtime=runtime)
        self.assertIs(module.serialized_parent_runtime(parent), runtime)
        del parent['training_runtime']
        parent['training_runtime_json'] = '{}'
        with self.assertRaises(KeyError):
            module.serialized_parent_runtime(parent)


    def test_source_binding_changes_only_with_admitted_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);(root / 'scripts').mkdir()
            (root / 'scripts/entry.py').write_text('value = 1')
            before = module.source_digest(root)
            (root / 'scripts/ignored.npz').write_bytes(b'output')
            self.assertEqual(module.source_digest(root), before)
            (root / 'scripts/entry.py').write_text('value = 2')
            self.assertNotEqual(module.source_digest(root), before)


if __name__ == '__main__':
    unittest.main()
