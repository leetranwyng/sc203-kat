from pathlib import Path
from collections import defaultdict,Counter
import ast,json,zipfile,random,hashlib
D=Path('notebooks/replay');D.mkdir(exist_ok=True)
def lines(z,n):return [json.loads(x) for x in z.open(n)]
a=zipfile.ZipFile('apc_3dsrbench_full_results.zip');p=zipfile.ZipFile('pcd_3dsrbench_full_results.zip')
old={}
for name in ('apc','pcd'):
 groups=defaultdict(list);by=defaultdict(list)
 if name=='apc':
  rows=lines(a,'apc_vis_predictions.jsonl');base={r['sample_id']:r for r in lines(a,'direct_predictions.jsonl')}
 else:rows=sum([lines(p,f'pcd_3dsrbench_full/predictions_gpu{i}.jsonl') for i in (0,1)],[])
 for r in rows:
  if name=='apc':
   sid=r['sample_id'];g=r['circular_group'];b=base[sid];hit=r['correct'];bh=b['correct'];bp=b['prediction'];task=r['paper_task']
  else:sid=r['qid'];g=r['base_qid'];hit=r['hit'];bh=r['baseline_hit'];bp=r['baseline_prediction'];task=r['category']
  groups[g].append(dict(id=sid,group=g,task=task,prediction=r['prediction'],baseline=bp,hit=bool(hit),baseline_hit=bool(bh),status=r.get('status','ok')))
 forced=set()
 for g,rs in groups.items():
  by[rs[0]['task'],all(x['baseline_hit'] for x in rs),all(x['hit'] for x in rs)].append(g)
  if any(x['status']!='ok' for x in rs):forced.add(g)
 rng=random.Random(42);selected=set(forced)
 for k in sorted(by):selected.update(rng.sample(sorted(by[k]),min(6,len(by[k]))))
 old[name]={x['id']:x for g in sorted(selected) for x in groups[g]}
 print(name,len(selected),len(old[name]),'forced errors',len(forced))
(D/'selected_cases.json').write_text(json.dumps(old,indent=1),encoding='utf8')
def src(n,i):return ''.join(n['cells'][i]['source'])
def setsrc(n,i,s):n['cells'][i]['source']=s.splitlines(True)
def save(n,name):
 for c in n['cells']:
  if c['cell_type']=='code':
   c.update(outputs=[],execution_count=None);s=''.join(c['source'])
   if s.startswith('%%writefile'):s=s.split('\n',1)[1]
   ast.parse(s)
 (D/name).write_text(json.dumps(n,ensure_ascii=False),encoding='utf8')
def rep(s,k,v):
 assert k in s,k
 return s.replace(k,v,1)
n=json.load(open('notebooks/apc_qwen25vl_3b_paper_tasks_kaggle.ipynb',encoding='utf8'))
setsrc(n,3,src(n,3).replace('apc_qwen3b_paper_tasks_visibility_v3','apc_step_c_replay')+"\nOLD="+repr(old['apc'])+"\nCFG['model_revision']='66285546d2b821cf421d4f5eb2576359d3770cd3'\nCFG['replay_ids']=sorted(OLD)\nCONFIG.write_text(json.dumps(CFG))")
mods={}
def edit(file,k,v):mods.setdefault(file,[]).append((k,v))
edit('scripts/apc_prepare_data.py','items=choose_samples(items,cfg)',"items=[r for r in choose_samples(items,cfg) if r['sample_id'] in cfg['replay_ids']]\n    assert len(items)==len(cfg['replay_ids'])")
edit('scripts/apc_experiment.py',"scratch=Path(cfg['work'])/'scratch'/hashlib.sha256((mode+row['sample_id']).encode()).hexdigest()[:16]","scratch=folder/'trace'/mode/row['sample_id']")
edit('scripts/apc_experiment.py','resolved=scratch.resolve();resolved.relative_to((Path(cfg[\'work\'])/\'scratch\').resolve());shutil.rmtree(resolved)',"pass  # Keep per-sample evidence")
edit('scripts/apc_experiment.py','if sanity:result.save','if True:result.save')
edit('scripts/apc_experiment.py',"image=image.convert('RGB')","image=image.convert('RGB')\n            os.environ['APC_TRACE_DIR']=str(scratch)\n            if mode=='apc_vis':image.save(scratch/'input.png')")
edit('scripts/apc_qwen_adapter.py','raw_output=raw[:240]','raw_output=raw,prompt=text')
f='apc/apc_pipeline.py'
edit(f,'import logging','import logging\nimport json\nfrom pathlib import Path')
edit(f,'for obj_name in objects_of_interest:\n            self.logger.info','for obj_index,obj_name in enumerate(objects_of_interest):\n            trace=Path(apc_args["trace_save_dir"])\n            prefix=trace / f"object_{obj_index}"\n            prefix.with_suffix(".name").write_text(obj_name)\n            self.logger.info')
edit(f,'if len(boxes) == 0:',"np.save(str(prefix)+'_boxes.npy', np.asarray([np.asarray(b.cpu() if hasattr(b,'cpu') else b) for b in boxes]))\n            if len(boxes) == 0:")
edit(f,'# largest mask',"""# largest mask
            Image.fromarray(np.asarray(segment_mask.cpu() if hasattr(segment_mask,'cpu') else segment_mask).astype('uint8')*255).save(str(prefix)+'_mask.png')
            np.save(str(prefix)+'_selected_box.npy',np.asarray(box2D))
            if not (trace/'depth.npy').exists():np.save(trace/'depth.npy',depth_npy)""")
edit(f,"return abstract_scene_tgt","Path(os.environ['APC_TRACE_DIR'],'scene.json').write_text(json.dumps(dict(viewer=ref_viewer,camera=abstract_scene_src,transformed=abstract_scene_tgt),default=lambda x:np.asarray(x).tolist()))\n        return abstract_scene_tgt")
# Patch scripts after setup, before data/worker launch. Idempotent on rerun.
edit('apc/vision_modules/detection.py','        boxes_sorted = sort_by_scores',"""        from pathlib import Path
        import json
        with (Path(os.environ['APC_TRACE_DIR'])/'detector.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(category=category,boxes=boxes.detach().cpu().tolist(),scores=scores.detach().cpu().tolist()))+'\\n')
        boxes_sorted = sort_by_scores""")
edit('scripts/apc_experiment.py',"message=str(exc)", "message=str(exc),traceback=traceback.format_exc(),failed_stage=state.get('stage')")
setsrc(n,0,"# Bước C · APC-Vis và baseline Qwen\nChạy lại các nhóm đã chọn từ kết quả cũ, giữ Visibility như lượt APC đã chạy. Tập này dùng phân tích lỗi. Chạy Run All với T4 ×2, Internet và secret HF_token.")
setsrc(n,12,"## Chạy các nhóm replay và lưu ảnh trung gian")
setsrc(n,9,'changes='+repr(mods)+'''
for rel, edits in changes.items():
    path=(WORK if rel.startswith('scripts/') else REPO)/rel
    script=path.read_text()
    for before,after in edits:
        if after not in script:
            assert before in script,(rel,before)
            script=script.replace(before,after,1)
    compile(script,str(path),'exec')
    path.write_text(script)
'''+src(n,9))
setsrc(n,11,src(n,11).replace('launch(smoke=True)\n','').removesuffix('launch(smoke=True)'))
setsrc(n,15,src(n,15).replace('apc_full_results.zip','apc_step_c_replay.zip').replace('import zipfile',"""current=[json.loads(line) for line in (OUT/'apc_vis_predictions.jsonl').read_text().splitlines()]
comparison=[dict(old=OLD[r['sample_id']], replay_prediction=r['prediction'],replay_status=r['status']) for r in current]
(OUT/'old_vs_replay.json').write_text(json.dumps(comparison))
import zipfile"""))
save(n,'APC_step_C.ipynb')
n=json.load(open(r'E:\Downloads\full-baseline-test.ipynb',encoding='utf8'))
setsrc(n,4,src(n,4).replace('/pcd_3dsrbench_full','/pcd_step_c_replay')+'\nOLD='+repr(old['pcd']))
setsrc(n,11,src(n,11).replace('selected_rows =','CONFIG["replay_ids"] = sorted(OLD)\nselected_rows ='))
s=src(n,9).replace('rows = all_rows\n','rows = [r for r in all_rows if r["qid"] in config["replay_ids"]]\n        assert len(rows)==len(config["replay_ids"])\n')
s=s.replace('if config["mode"] == "sanity" and scores is None:', 'if scores is None:').replace('if config["mode"] == "sanity":\n                    record.update','if True:\n                    record.update')
s=s.replace('                torch.cuda.synchronize()\n                record', '''                if camera_raw is None:
                    camera_raw = reference.generate_answer(image,camera_prompt,model,processor,device)
                record["diagnostic_camera_prediction"]=parse_letter(camera_raw,labels)[0]
                record["diagnostic_camera_raw"]=camera_raw
                record["diagnostic_contrastive_prediction"]=max(labels,key=lambda label:scores[label]["contrastive"])
                record["extraction_prompt"],record["extraction_raw"]=reference.extraction_trace[extraction_key]
                torch.cuda.synchronize()
                record''')
setsrc(n,9,s);s=src(n,8).replace('import argparse', 'import argparse\nextraction_trace = {}').replace('        json_match = re.search','        extraction_trace[category,question]=(prompt,generated_text)\n        json_match = re.search',1);setsrc(n,8,s)
setsrc(n,17,src(n,17).replace('import zipfile','''(OUTPUT_DIR/'old_vs_replay.json').write_text(json.dumps([dict(old=OLD[r['qid']],replay_prediction=r['prediction'],replay_branch=r['branch']) for r in records]))
import zipfile''').replace(' and "images" not in path.relative_to(OUTPUT_DIR).parts',''))
setsrc(n,0,'# Bước C · PCD và baseline Qwen\nChạy các nhóm đã chọn với Front/Behind gốc. Kết quả hai nhánh đối chứng được lưu riêng. Chạy Run All với T4 ×2 và Internet.')
save(n,'PCD_step_C.ipynb')
