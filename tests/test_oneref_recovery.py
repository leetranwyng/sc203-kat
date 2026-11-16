import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('oneref_recovery',ROOT/'scripts/recover_apc_oneref.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class OneRefRecoveryTests(unittest.TestCase):
    def test_recovery_keeps_completed_branches_and_original_identity(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output';out.mkdir()
            py=root/'extra_envs/oneref/bin/python';py.parent.mkdir(parents=True);py.touch()
            cfg=dict(output_dir=str(out),variants=['sam3','oneref','oa2'],repos={'oneref':str(root/'oneref')},weights={'oneref':'cached.pth'})
            (root/'ablation.json').write_text(json.dumps(cfg))
            context=dict(py='apc-python',uv='uv',env=dict(PYTHONPATH='old-code/src',
                CUDA_VISIBLE_DEVICES='GPU-0,GPU-1',LD_LIBRARY_PATH='apc-cuda'))
            (root/'apc_context.json').write_text(json.dumps(context))
            (out/'summary.json').write_text(json.dumps(dict(technical_pass=True)))
            (out/'identity.json').write_text('original identity')
            calls=[]
            def run(args,**kwargs):
                calls.append((args,kwargs));return subprocess.CompletedProcess(args,0,stdout='setuptools==75.8.0\n')
            module.recover(root,env=dict(PYTHONPATH='old-code/src',LD_LIBRARY_PATH='driver-only'),run=run)
            retry=json.loads((root/'oneref_recovery_config.json').read_text())
            self.assertEqual(retry['variants'],['oneref']);self.assertTrue(retry['retry_errors'])
            self.assertEqual(json.loads((root/'ablation.json').read_text()),cfg)
            self.assertEqual((out/'identity.json').read_text(),'original identity')
            phases=[(a[3],a[-1]) for a,kw in calls if '-m' in a]
            self.assertEqual(phases,[('sc203_kat.ablation.perception','oneref'),
                                     ('sc203_kat.ablation.apc_worker','geometry'),
                                     ('sc203_kat.ablation.apc_worker','answers')])
            perception=next(kw for a,kw in calls if 'sc203_kat.ablation.perception' in a)
            self.assertEqual(perception['env']['CUDA_VISIBLE_DEVICES'],'GPU-0')
            self.assertEqual(perception['env']['LD_LIBRARY_PATH'],'driver-only')
            self.assertFalse(any('setup_apc' in str(a) or 'sc203_kat.ablation.assets' in a for a,kw in calls))

    def test_setup_probe_failure_resumes_original_runner_without_native_setup(self):
        import types,sys
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);out=root/'output';out.mkdir()
            py=root/'extra_envs/oneref/bin/python';py.parent.mkdir(parents=True);py.touch()
            cfg=dict(output_dir=str(out),variants=['sam3','oneref','oa2'])
            (root/'ablation.json').write_text(json.dumps(cfg))
            context=dict(py='apc-python',uv='uv',env=dict(PYTHONPATH='old-code/src',CUDA_VISIBLE_DEVICES='GPU-0,GPU-1'))
            (root/'apc_context.json').write_text(json.dumps(context))
            calls=[];resumed=[]
            package=types.ModuleType('sc203_kat');package.__file__=str(root/'code/src/sc203_kat/__init__.py')
            ablation=types.ModuleType('sc203_kat.ablation')
            setup=types.SimpleNamespace(setup_apc=lambda *a:None)
            def resume(actual_cfg,code_root):
                self.assertEqual(setup.setup_apc(actual_cfg,code_root),context)
                self.assertTrue(actual_cfg['retry_errors']);resumed.append(actual_cfg)
            ablation.run=types.SimpleNamespace(run=resume);ablation.setup=setup
            def run(args,**kwargs):
                calls.append(args)
                if len(args)>3 and args[:3]==['apc-python','-u','-c']:
                    with patch.dict(sys.modules,{'sc203_kat':package,'sc203_kat.ablation':ablation}), patch.object(sys,'argv',['recover',str(root)]):
                        exec(args[3],{})
                return subprocess.CompletedProcess(args,0,stdout='click==8.1.8\n')
            module.recover(root,env={},run=run)
            self.assertEqual(len(resumed),1)
            self.assertEqual(json.loads((root/'ablation.json').read_text()),cfg)
            self.assertFalse(any('sc203_kat.ablation.apc_worker' in c for c in calls))

if __name__=='__main__':unittest.main()
