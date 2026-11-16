import math
from pathlib import Path
import tempfile
import unittest
from sc203_kat.ablation.common import (oa2_direction,select_oneref,unpad_oneref,
    cxcywh_from_xyxy,xyxy_from_cxcywh,inside,write,read)
from sc203_kat.ablation.report import summarize
from sc203_kat.ablation.remote_zip import extract_mixup

class GeometryTests(unittest.TestCase):
    def test_orientation_zero_roll_matches_apc_v1(self):
        for a in range(0,360,15):
            for e in (-89,-45,0,45,89):
                aa,ee=map(math.radians,(a,e))
                expected=[-math.sin(aa),math.sin(ee)*math.cos(aa),-math.cos(ee)*math.cos(aa)]
                for x,y in zip(expected,oa2_direction(a,e,0)):self.assertAlmostEqual(x,y,places=12)
    def test_roll_rotates_image_plane(self):
        v=oa2_direction(90,0,90)
        self.assertAlmostEqual(v[0],0);self.assertAlmostEqual(v[1],1);self.assertAlmostEqual(v[2],0)
    def test_orientation_unit_length(self):
        for a,e,r in [(12,65,90),(359,-20,-120),(90,90,180)]:
            self.assertAlmostEqual(sum(v*v for v in oa2_direction(a,e,r)),1)
    def test_select_instance_not_first(self):
        selected,scores=select_oneref([[0,0,10,10],[50,40,90,80]],[51,41,89,79])
        self.assertEqual(selected,1);self.assertGreater(scores[1],.8)
    def test_no_overlap_is_error_not_first(self):
        with self.assertRaises(ValueError):select_oneref([[0,0,10,10]],[20,20,30,30])
    def test_letterbox_coordinates(self):
        # 640x320 -> 384x192, top padding 96. Centered object in native image.
        box=[160,80,480,240];network=[96,144,288,240]
        recovered=unpad_oneref(cxcywh_from_xyxy(network,384,384),640,320)
        for a,b in zip(box,recovered):self.assertAlmostEqual(a,b)
    def test_box_conversion(self):
        box=[32,48,160,240]
        recovered=xyxy_from_cxcywh(cxcywh_from_xyxy(box,640,480),640,480)
        # Preserve APC's integer truncation policy; normalized float round trips
        # can shift an edge by one pixel. Do not silently change the baseline.
        self.assertTrue(all(abs(a-b)<=1 for a,b in zip(recovered,box)))
    def test_paths_cannot_escape(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):inside(t,'../escape')
    def test_nan_cannot_be_persisted(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):write(Path(t)/'x.json',dict(x=float('nan')))

class ScoringTests(unittest.TestCase):
    def test_errors_wrong_and_missing_not_scored_as_completed(self):
        items=[dict(sample_id=str(i),paper_task='visibility') for i in range(3)]
        records={('baseline','0'):dict(correct=True,status='ok'),
                 ('baseline','1'):dict(correct=False,status='error'),
                 ('baseline','2'):dict(correct=False,status='invalid'),
                 ('oa2','0'):dict(correct=False,status='ok'),
                 ('oa2','1'):dict(correct=True,status='ok')}
        table,pairs=summarize(items,records,['baseline','oa2'])
        self.assertEqual(table[0]['wrong'],2);self.assertEqual(table[0]['accuracy'],1/3)
        self.assertIsNone(table[1]['accuracy']);self.assertEqual(table[1]['unattempted'],1)
        self.assertEqual(pairs[0]['a_only'],1);self.assertEqual(pairs[0]['b_only'],1)
        self.assertEqual(pairs[0]['unpaired'],1)

    def test_historical_baseline_uses_saved_answer_without_render_job(self):
        from sc203_kat.ablation.report import build
        import json
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            write(out/'identity.json',dict(variants=['sam3','oa2']))
            (out/'samples.jsonl').write_text(json.dumps(dict(sample_id='1',paper_task='facing',
                question='Which object?',choices={'A':'cat','B':'dog'},answer='A',
                historical_status='ok',historical_prediction='B'))+'\n')
            (out/'original/1').mkdir(parents=True)
            for variant,status,prediction in [('sam3','ok','A'),('oa2','error',None)]:
                write(out/f'answers/{variant}/1/result.json',dict(status=status,prediction=prediction,
                    attempt=f'answers/{variant}/1/attempt_001'))
            build(out);value=read(out/'summary.json')
            self.assertEqual(value['baseline_source'],'historical saved result; never rerun')
            table={r['variant']:r for r in value['by_task']}
            self.assertEqual(table['baseline']['accuracy'],0)
            self.assertEqual(table['sam3']['accuracy'],1)
            self.assertEqual(table['oa2']['wrong'],1)
            self.assertFalse(value['technical_pass'])
            self.assertFalse((out/'answers/baseline').exists())

class RangeExtractionTests(unittest.TestCase):
    def test_extract_only_mixup_with_crc_validation(self):
        import io,zipfile
        from unittest.mock import patch
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('release/mixup/best_checkpoint.pth',b'mixup-checkpoint'*100)
            z.writestr('release/flickr/best_checkpoint.pth',b'unused')
        with tempfile.TemporaryDirectory() as t:
            with patch('sc203_kat.ablation.remote_zip.HTTPRangeFile',return_value=io.BytesIO(data.getvalue())):
                target=Path(t)/'checkpoint.pth';extract_mixup('https://example.test/checkpoint.zip',target)
            self.assertEqual(target.read_bytes(),b'mixup-checkpoint'*100)
            self.assertEqual(len(list(Path(t).iterdir())),1)

class MergeTests(unittest.TestCase):
    def test_merge_preserves_three_branches_and_historical_baseline(self):
        import json
        from sc203_kat.ablation.merge import merge
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sources=[]
            for name,variants in [('one',['sam3','oneref']),('two',['oa2'])]:
                out=root/name;out.mkdir();sources.append(out)
                write(out/'identity.json',dict(variants=variants,source_sha='same',sample_ids=['1']))
                (out/'original/1').mkdir(parents=True)
                (out/'samples.jsonl').write_text(json.dumps(dict(sample_id='1',paper_task='facing',
                    question='Q',choices={'A':'yes','B':'no'},answer='A',historical_status='ok',
                    historical_prediction='B'))+'\n')
                for variant in variants:
                    write(out/f'answers/{variant}/1/result.json',dict(status='ok',prediction='A',
                        attempt=f'answers/{variant}/1/attempt_001'))
            page=merge(sources,root/'merged')
            result=read(page.parent/'summary.json')
            self.assertEqual(len(result['by_task']),4)
            self.assertEqual(len(result['pairs']),6)
            self.assertEqual(read(page.parent/'identity.json')['variants'],['sam3','oneref','oa2'])

if __name__=='__main__':unittest.main()
