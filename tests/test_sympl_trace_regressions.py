import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/sc203_kat/sympl_trace/scripts'),str(ROOT/'profiles/sympl/overlay/scripts')]
import trace_runtime as tr
from analyze120 import build_gallery
from sc203_kat.sympl_trace.__main__ import report
from sc203_kat.sympl_trace.instrumentation import instrument
from PIL import Image
import numpy as np


class Regressions(unittest.TestCase):
    def test_nested_image_and_array_keys_from_real_hook_shape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'traces/5486/attempt_001'
            t=tr.Trace(root,folder)
            original={'image':Image.new('RGB',(4,4)),'other':{'image':None},
                      'array':{'array':np.arange(300)},'text':'<script>unsafe</script>'}
            packed=t.pack(original)
            restored=tr.unpack(packed,root)
            self.assertIsInstance(restored['image'],Image.Image)
            np.testing.assert_array_equal(restored['array']['array'],np.arange(300))
            t.event(phase='start',id=1,component='get_objects_of_interest',inputs=packed)
            t.event(phase='error',id=1,component='get_objects_of_interest',error='fixture error')
            row=dict(sample_id='5486',attempt_path=folder.relative_to(root).as_posix(),paper_task='left_right',
                apc={'correct':True,'prediction':'A'},correct=False,source_question='q',question='q',choices={'A':'left'},answer='A',prediction=None,raw_output=None)
            build_gallery(root,[row],[dict(root_cause='runtime_error',certainty='observed',flags=[])],{})
            html=(root/'gallery.html').read_text(encoding='utf8')
            self.assertIn('fixture error',html)
            self.assertIn(packed['image']['image'],html)
            self.assertNotIn('<script>unsafe</script>',html)
            self.assertFalse(tr.artifact_ref({'image':'javascript:alert(1)'},'image'))

    def test_report_keeps_worker_and_analysis_exit_and_runs_sanity(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(output=d,work=d,repo=d)
            with patch('sc203_kat.sympl_trace.__main__.subprocess.run',side_effect=[
                subprocess.CompletedProcess([],7),subprocess.CompletedProcess([],1)]) as calls:
                state=report(cfg,'python',{},worker_exit=9,sanity=True)
            self.assertEqual(state,dict(worker_exit=9,analysis_exit=7,sanity_exit=1))
            self.assertEqual(calls.call_count,2)
            self.assertTrue((Path(d)/'execution_status.json').exists())

    def test_instrumentation_cannot_modify_shared_profile(self):
        source=ROOT/'profiles/sympl/overlay/repo'
        before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*.py')}
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)/'repo';shutil.copytree(source,repo)
            instrument(repo)
            import ast, types
            original=source/'sympl/vision_modules/depth.py'
            results=[]
            for p in (original,repo/'sympl/vision_modules/depth.py'):
                tree=ast.parse(p.read_text(encoding='utf8'))
                f=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='unproject_to_3D_from_bbox')
                o3d=types.SimpleNamespace(geometry=types.SimpleNamespace(Image=lambda x:x,
                    RGBDImage=types.SimpleNamespace(create_from_color_and_depth=lambda **kw:types.SimpleNamespace(depth=kw['depth']))))
                ns=dict(np=np,Image=Image,o3d=o3d)
                exec(compile(ast.Module(body=[f],type_ignores=[]),'depth','exec'),ns)
                depth=np.full((32,40),3.,dtype=np.float32);depth[:5]=7
                K=np.array([[40,0,20],[0,40,16],[0,0,1]],float)
                results.append(ns[f.name](None,Image.new('RGB',(40,32)),depth,[1,1,38,30],K))
            np.testing.assert_array_equal(*results)
        self.assertEqual(before,{str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*.py')})

    def test_notebooks_are_launchers_and_outputs_cleared(self):
        import ast
        paths=list((ROOT/'notebooks/sympl').glob('SymPL_7B_APC120_*_kaggle.ipynb'))
        self.assertEqual(len(paths),3)
        for p in paths:
            self.assertLess(p.stat().st_size,12000)
            nb=json.loads(p.read_text(encoding='utf8'))
            for c in nb['cells']:
                if c['cell_type']=='code':
                    text=''.join(c['source']);ast.parse(text)
                    self.assertNotIn('base64.b64decode',text)
                    self.assertEqual(c['outputs'],[])

    def test_failed_smoke_blocks_full_and_still_archives(self):
        from sc203_kat.sympl_trace import __main__ as runner
        with tempfile.TemporaryDirectory() as d:
            work=Path(d)/'work';out=Path(d)/'out'
            def fake_stage(name,ns):
                if name=='01_prepare.py':
                    (work/'repo').mkdir(parents=True);out.mkdir()
                    ns.update(CFG=dict(work=str(work),output=str(out),repo=str(work/'repo'),model_revision='pin'),CONFIG=work/'config.json')
                elif name=='02_environment.py':ns.update(PY=sys.executable,ENV={},UV='uv')
                elif name=='03_native_assets.py':ns['CFG']['model_sha']='pin'
            with patch.object(runner.platform,'system',return_value='Linux'), \
                 patch('sc203_kat.runtime.kaggle.ensure_submodule'), \
                 patch('sc203_kat.runtime.kaggle.source_metadata',return_value={'commit':'test'}), \
                 patch.object(runner,'stage',side_effect=fake_stage), \
                 patch.object(runner,'instrument'), \
                 patch('shared_trace_input.preflight'), \
                 patch.object(runner.subprocess,'check_output',return_value='frozen'), \
                 patch.object(runner,'worker',return_value=False) as worker, \
                 patch.object(runner,'archive_output') as archive:
                with self.assertRaisesRegex(RuntimeError,'full 120 did not start'):
                    runner.run(dict(mode='full',runtime_dir=str(work),output_dir=str(out)))
                self.assertEqual(worker.call_count,1)
                self.assertTrue(worker.call_args.args[0]['smoke_only'])
                self.assertTrue(worker.call_args.args[0]['sanity'])
                archive.assert_called_once_with(out)
                self.assertTrue((out/'launcher_error.txt').exists())

if __name__=='__main__':unittest.main()
