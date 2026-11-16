import json, sys, tempfile, types, unittest, zipfile, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'src/sc203_kat/sympl_trace/scripts'
sys.path[:0]=[str(SCRIPTS),str(ROOT/'profiles/sympl/overlay/scripts')]
import trace_runtime as tr
from analyze120 import renderer,layout_decision,analyze_one,missing_refs,build_gallery

class TraceTests(unittest.TestCase):

    def test_roundtrip_and_nested_events(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            t = tr.Trace(root, root / 'sample')
            tr.ACTIVE = t
            a = np.arange(3000, dtype=np.float32).reshape(30, 100)
            self.assertTrue(np.array_equal(a, tr.unpack(t.pack(a), root)))
            self.assertTrue(np.array_equal(np.asarray(Image.new('RGB', (4, 4), 'red')), np.asarray(tr.unpack(t.pack(Image.new('RGB', (4, 4), 'red')), root))))
            owner = types.SimpleNamespace(inner=lambda x: x * 2)
            t.hook(owner, 'inner')
            owner.outer = lambda x: owner.inner(x) + 1
            t.hook(owner, 'outer')
            self.assertEqual(owner.outer(5), 11)
            e = [json.loads(s) for s in (root / 'sample/calls.jsonl').read_text().splitlines()]
            self.assertEqual(e[1]['parent'], e[0]['id'])
            self.assertEqual(missing_refs(root, root / 'sample'), [])
            tr.ACTIVE = None

    def test_real_renderer_and_ambiguous_boundaries(self):
        render = renderer(ROOT / 'profiles/sympl/overlay/repo')
        spec = [dict(type='rectangle', coords=[[0, 0], [511, 256]], color='yellow')]
        bg = render(np.array([]), np.array([]), np.empty((0, 3)), spec)
        row = dict(paper_task='visibility', choices={'A': 'yes', 'B': 'no'})
        mapping = {'ref': 'yellow', 'board': 'red'}
        x = layout_decision([300], [100], [], mapping, bg, row)
        self.assertEqual(x['expected_option'], 'A')
        self.assertEqual(x['boundary_overlap'], [False])
        self.assertEqual(layout_decision([300], [256], [], mapping, bg, row)['boundary_overlap'], [True])
        mapping = {'ref': 'yellow', 'cat': 'red', 'dog': 'blue'}
        row = dict(paper_task='facing', choices={'A': 'cat', 'B': 'dog'})
        self.assertIsNone(layout_decision([100, 300], [100, 100], [], mapping, bg, row)['expected_option'])

    def make_case(self, root, sid, raw, pred, gt):
        folder = root / 'traces' / sid / 'attempt_001'
        t = tr.Trace(root, folder)
        row = dict(sample_id=sid, attempt_path=folder.relative_to(root).as_posix(), paper_task='visibility', choices={'A': 'yes', 'B': 'no'}, prediction=pred, answer=gt, correct=pred == gt, apc={'correct': False, 'prediction': 'B', 'previews': []}, status='ok', raw_output='Yes, Visible', source_question='Where is board?', question='Is board visible?')
        Image.new('RGB', (64, 64), 'white').save(folder / 'input.png')
        u = np.array([300.0])
        v = np.array([100.0])
        colors = np.array([[255, 0, 0]])
        mapping = {'ref': 'yellow', 'board': 'red'}
        spec = [dict(type='rectangle', coords=[[0, 0], [511, 256]], color='yellow')]
        render = renderer(ROOT / 'profiles/sympl/overlay/repo')
        actual = render(u, v, colors, spec)
        t.event(phase='end', id=1, component='do_abstraction', result=t.pack([u, v, {}, [], colors, ['board'], mapping, [0, 0, 0], True]))
        t.event(phase='end', id=2, component='do_bipartition', result=t.pack([spec, u, v]))
        t.event(phase='end', id=3, component='do_localization', result=t.pack(actual))
        t.event(phase='start', id=4, parent=None, component='symbolic_layout_prompting', inputs={})
        t.event(phase='start', id=5, parent=4, component='process_messages', inputs={})
        t.event(phase='end', id=5, component='process_messages', result=raw)
        t.event(phase='end', id=4, component='symbolic_layout_prompting', result='Yes, Visible')
        return row

    def test_failure_attribution_and_gallery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sid, raw, pred, gt, cause, flag in [('vlm', 'Black', 'B', 'A', 'vlm_layout_reading', 'vlm_reading_disagrees_with_layout'), ('parser', 'Yellow', 'B', 'A', 'parser_or_mapping', 'parser_mapping_disagreement'), ('upstream', 'Yellow', 'A', 'B', 'upstream_geometry_or_partition', 'layout_disagrees_with_gt_upstream_review'), ('compensation', 'Black', 'B', 'B', 'vlm_layout_reading', 'correct_answer_with_compensating_errors')]:
                row = self.make_case(root, sid, raw, pred, gt)
                diag = analyze_one(root, row, ROOT / 'profiles/sympl/overlay/repo')
                self.assertEqual(diag['root_cause'], cause)
                self.assertIn(flag, diag['flags'])
                self.assertTrue(diag['integrity']['renderer_replay_exact'])
            build_gallery(root, [row], [diag], {'completed': 1})
            self.assertIn('Manual review', (root / 'gallery.html').read_text(encoding='utf8'))

    def test_directory_discovery_and_corruption(self):
        import hashlib
        from shared_trace_input import TraceBundle, find_bundle
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            dataset = base / 'datasets/user/sc203-apc-trace120/v1/sc203_apc_trace120'
            dataset.mkdir(parents=True)
            rows = []
            hashes = {}
            for i in range(120):
                sid = str(i)
                rel = f'original/{sid}/input.png'
                p = dataset / rel
                p.parent.mkdir(parents=True)
                Image.new('RGB', (4, 4), (i, 0, 0)).save(p)
                hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append(dict(sample_id=sid, paper_task=('left_right', 'visibility', 'facing')[i // 40], image_path=rel, trace_path=f'original/{sid}', answer='A', choices={'A': 'yes', 'B': 'no'}))
            (dataset / 'samples.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in rows)))
            (dataset / 'bundle.json').write_text(json.dumps(dict(count=120, files=hashes)))
            self.assertEqual(find_bundle(None, input_root=base), dataset.resolve())
            with TraceBundle(dataset) as bundle:
                self.assertEqual(bundle.validate()['rows'], 120)
            with zipfile.ZipFile(base / 'sc203_apc_trace120.zip', 'w') as z:
                for p in dataset.rglob('*'):
                    if p.is_file():
                        z.write(p, 'sc203_apc_trace120/' + p.relative_to(dataset).as_posix())
            self.assertEqual(find_bundle(None, input_root=base), dataset.resolve())
            (dataset / 'original/0/input.png').write_bytes(b'corrupt')
            with TraceBundle(dataset) as bundle:
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    bundle.validate()
            with self.assertRaisesRegex(ValueError, 'TRACE_INPUT'):
                find_bundle(None, input_root=base / 'missing')

    def test_sanity_gate_is_not_an_accuracy_gate(self):
        from sanity120 import check
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(check(root)['status'], 'FAIL')
            rows = []
            valid = {}
            for i, task in enumerate(['left_right'] * 2 + ['visibility'] * 2 + ['facing'] * 2):
                row = self.make_case(root, str(i), 'Yellow', 'A', 'B')
                row.update(paper_task=task, trace_closed=True)
                rows.append(row)
                tr.atomic_json(root / 'traces' / str(i) / 'result.json', row)
                valid[str(i)] = dict(missing_stages=[], missing_artifacts=[], unterminated_calls=[], renderer_replay_exact=True)
            (root / 'samples.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in rows)))
            tr.atomic_json(root / 'trace_validation.json', valid)
            tr.atomic_json(root / 'failure_cases.json', [dict(sample_id=r['sample_id'], root_cause='upstream_geometry_or_partition') for r in rows])
            tr.atomic_json(root / 'input_validation.json', dict(status='PASS'))
            (root / 'gallery.html').write_text('Synthetic gate fixture')
            report = check(root)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['correct'], 0)
            self.assertEqual(check(root, worker_exit=1)['status'], 'FAIL')
            artifact = next((root / 'artifacts').iterdir())
            artifact.write_bytes(b'corrupt')
            report = check(root)
            self.assertEqual(report['status'], 'FAIL')
            self.assertFalse(report['checks']['artifact_content_hashes_match'])
