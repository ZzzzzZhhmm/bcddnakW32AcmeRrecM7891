"""Invalid saved-prefix requests must fail before importing the model stack."""
from pathlib import Path
import importlib.util,json,os,subprocess,sys,tempfile,unittest


ROOT=Path(__file__).resolve().parents[1]
PROGRAM=ROOT/'scripts/preflight_rebuttal_legacy_policy.py'
SPEC=importlib.util.spec_from_file_location('legacy_checkpoint_cli_under_test',PROGRAM)
CLI=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CLI)


class SavedPrefixBoundaryTests(unittest.TestCase):
    def rejected_before_torch(self,extra,expected):
        with tempfile.TemporaryDirectory(prefix='warm-legacy-cli-') as directory:
            owned=Path(directory).resolve()
            self.assertTrue(owned.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            (owned/'torch.py').write_text("raise AssertionError('Unexpected model-stack import before admission')\n")
            output=owned/'output'
            env=dict(os.environ,PYTHONPATH=str(owned),CUDA_VISIBLE_DEVICES='')
            command=[sys.executable,str(PROGRAM),'--recipe',str(owned/'recipe.yaml'),
                '--sealed-dir',str(owned/'sealed'),'--infer-kwargs',str(owned/'infer.pt'),
                '--model-root',str(owned/'models'),'--output',str(output),*extra]
            result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,1,(result.stdout,result.stderr))
            state=json.loads((output/'operator.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertIn(expected,state['error'])
            self.assertNotIn('Unexpected model-stack import',state['traceback'])
            self.assertEqual(state['optimizer_updates'],0)
            self.assertEqual(state['actual_policy_generations'],0)
            self.assertEqual(state['policy_generation_attempts'],0)

    def test_identity_without_saved_checkpoint(self):
        self.rejected_before_torch(['--expected-checkpoint-sha256','a'*64],
            'checkpoint identity requires an explicit saved checkpoint')

    def test_invalid_saved_identity(self):
        self.rejected_before_torch(['--checkpoint','not-read.pt','--expected-checkpoint-sha256','g'*64,
            '--expected-checkpoint-update','640'],'exact SHA256 and positive optimizer update')

    def test_trained_model_cannot_reuse_untrained_output(self):
        self.rejected_before_torch(['--checkpoint','not-read.pt','--expected-checkpoint-sha256','a'*64,
            '--expected-checkpoint-update','640','--resume-qualification-from','not-read'],
            'cannot reuse an untrained sampler output')


class StrictPublicLoaderTests(unittest.TestCase):
    def test_path_only_public_entrypoint_grants_expected_identity(self):
        digest='b'*64;path=Path('recorded-prefix.pt')
        class PublicWARM:
            def load_checkpoint(self,checkpoint,optimizer=None):
                self.path=checkpoint;self._warm_loaded_checkpoint_sha256=digest
        model=PublicWARM()
        CLI._restore_declared_warm_checkpoint(model,path,digest)
        self.assertEqual(model.path,path)

    def test_ungranted_identity_is_rejected(self):
        class RejectedWARM:
            def load_checkpoint(self,checkpoint,optimizer=None):self._warm_loaded_checkpoint_sha256=None
        with self.assertRaisesRegex(ValueError,'did not grant'):
            CLI._restore_declared_warm_checkpoint(RejectedWARM(),Path('not-read.pt'),'b'*64)

    def test_original_loader_failure_is_preserved(self):
        class InvalidCheckpointWARM:
            def load_checkpoint(self,checkpoint,optimizer=None):raise ValueError('Original checkpoint preflight rejected')
        with self.assertRaisesRegex(ValueError,'Original checkpoint preflight rejected'):
            CLI._restore_declared_warm_checkpoint(InvalidCheckpointWARM(),Path('not-read.pt'),'b'*64)


if __name__=='__main__':unittest.main()
