import hashlib
import html
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from sc203_kat.ablation.portable import safe_component,path_map,rewrite_payload,patch_apc_filenames,SAFE
from sc203_kat.ablation.bundle import repair_zip

class PortableTraceTests(unittest.TestCase):
    def test_safe_distinct_idempotent_names(self):
        values=["detection_candidates_man's legs.png",'detection_candidates_man_s_legs.png',
                'person/red.png','tủ lạnh.png','.', '..','CON','image.png','a'*300+'.png']
        names=[safe_component(v) for v in values]
        self.assertEqual(len(set(names)),len(names))
        for name in names:
            self.assertRegex(name,SAFE);self.assertLessEqual(len(name),180)
            self.assertEqual(safe_component(name),name)

    def test_html_and_json_links_change_but_semantic_text_does_not(self):
        image="original/1/detection_candidates_man's legs.png"
        names=[image,'original/1/viewer.html','original/1/vlm/001.json']
        mapping=path_map(names)
        data=('<p>man&#x27;s legs</p><img src="detection_candidates_man&#x27;s legs.png">').encode()
        result=rewrite_payload(data,names[1],mapping).decode()
        self.assertIn(Path(mapping[image]).name,result)
        self.assertIn('<p>man&#x27;s legs</p>',result)
        value={'category':"man's legs",'prompt':"Select man's legs, then answer A.",
               'images':["../detection_candidates_man's legs.png"]}
        updated=json.loads(rewrite_payload(json.dumps(value).encode(),names[2],mapping))
        self.assertEqual(updated['category'],value['category'])
        self.assertEqual(updated['prompt'],value['prompt'])
        self.assertEqual(updated['images'],['../'+Path(mapping[image]).name])

    def test_only_artifact_path_is_patched_idempotently(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);folder=root/'apc/vision_modules';folder.mkdir(parents=True)
            for module,prefix in [('detection','detection_candidates'),('orientation','orientation')]:
                (folder/f'{module}.py').write_text(f'category="man\x27s legs"\nprompt=category\npath=f"{prefix}_{{category}}.png"\n')
            first=patch_apc_filenames(root);second=patch_apc_filenames(root)
            self.assertEqual(first,second)
            source=(folder/'detection.py').read_text()
            self.assertIn('prompt=category',source);self.assertIn('{safe_component(category)}',source)

    def test_repair_rebuilds_manifest_keeps_binary_and_gold(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old=root/'old.zip';new=root/'new.zip';prefix='trace120'
            image="original/1/detection_candidates_man's legs.png"
            payload={image:b'unchanged-image-bytes','original/1/input.png':b'input-image',
                     'original/1/viewer.html':b'<img src="detection_candidates_man&#x27;s legs.png">'}
            sample=dict(sample_id='1',paper_task='facing',question="Where are man's legs?",answer='A',
                choices={'A':"man's legs",'B':'box'},image_path='original/1/input.png',trace_path='original/1')
            payload['samples.jsonl']=(json.dumps(sample)+'\n').encode()
            metadata=dict(schema=1,files={p:hashlib.sha256(v).hexdigest() for p,v in payload.items()})
            with zipfile.ZipFile(old,'w') as z:
                for p,v in payload.items():z.writestr(prefix+'/'+p,v)
                z.writestr(prefix+'/bundle.json',json.dumps(metadata))
            result=repair_zip(old,new);self.assertEqual(result['renamed_files'],1)
            with zipfile.ZipFile(new) as z:
                self.assertIsNone(z.testzip())
                manifest=json.loads(z.read(prefix+'/bundle.json'))
                mapping=json.loads(z.read(prefix+'/filename_map.json'))['renamed_files']
                for name,digest in manifest['files'].items():
                    self.assertEqual(hashlib.sha256(z.read(prefix+'/'+name)).hexdigest(),digest)
                    for part in Path(name).parts:self.assertRegex(part,SAFE)
                self.assertEqual(z.read(prefix+'/'+mapping[image]),payload[image])
                self.assertEqual(json.loads(z.read(prefix+'/samples.jsonl')),sample)
                self.assertIn(Path(mapping[image]).name,z.read(prefix+'/original/1/viewer.html').decode())

if __name__=='__main__':unittest.main()
