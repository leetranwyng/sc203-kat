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
            cfg=dict(output_dir=str(out),variants=['sam3','oneref','oa2'],repos={'oneref':str(root/'oneref')})
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

if __name__=='__main__':unittest.main()
