import collections, hashlib, json, pathlib
P=pathlib.Path(__file__).parent
C=json.loads((P/'cases.json').read_text(encoding='utf8'))
M=['APC-Vis','PCD','SymPL','VST','Spatial-SSRL']
def write(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
conf=[];flip=[]
for m in M:
 for task in ['left_right','front_behind','visibility','facing']:
  rr=[r for r in C if r['task']==task and m in r['models'] and r['models'][m]['state']!='Missing']
  if not rr:continue
  cells=collections.Counter((r['choices'][r['answer']],r['choices'].get(r['models'][m]['prediction'],'[invalid/error]')) for r in rr)
  conf.extend(dict(model=m,task=task,gold=g,predicted=p,count=n) for (g,p),n in cells.items())
  for isflip in [False,True]:
   ss=[r for r in rr if ('-flip' in r['group'])==isflip]
   flip.append(dict(model=m,task=task,is_flip=isflip,rows=len(ss),correct=sum(r['models'][m]['state']=='Correct' for r in ss)))
write('confusion.json',conf);write('flip_slices.json',flip)

grouped=collections.defaultdict(list)
for c in C:grouped[(c['task'],c['group'])].append(c)
queue={}
def pick(label,test,limit):
 count=0;images=set()
 for k,rr in sorted(grouped.items()):
  if not any(test(c) for c in rr):continue
  r=rr[0]
  if r['image'] in images:continue
  images.add(r['image'])
  if k not in queue:queue[k]=dict(task=k[0],group=k[1],sample_ids=[c['id'] for c in rr],question=r['question'],image=r['image'],priorities=[],review=dict(viewer_correct=None,object_identity_correct=None,box_correct=None,orientation_correct=None,depth_correct=None,frame_transform_correct=None,final_answer_mapping_correct=None,small_object=None,occlusion=None,multiple_similar_instances=None,planar_or_symmetric_viewer=None,body_head_disagreement=None,camera_roll=None,notes='',evidence='',certainty='unreviewed'))
  queue[k]['priorities'].append(label);count+=1
  if count>=limit:break
pick('All five wrong: inspect image geometry first',lambda c:c['task']=='left_right' and all(c['models'].get(m,{}).get('state')=='Wrong' for m in M),100)
pick('APC technical failure',lambda c:c['models'].get('APC-Vis',{}).get('state') in ['Invalid','Error'],100)
pick('PCD selected branch wrong; alternative correct',lambda c:'alternative correct' in c['models'].get('PCD',{}).get('label',''),25)
pick('PCD both branches wrong',lambda c:'Both PCD branches wrong' in c['models'].get('PCD',{}).get('label',''),100)
pick('VST Left/Right error',lambda c:c['task']=='left_right' and c['models'].get('VST',{}).get('state')=='Wrong',25)
pick('SSRL wrong but APC correct',lambda c:c['models'].get('Spatial-SSRL',{}).get('state')=='Wrong' and c['models'].get('APC-Vis',{}).get('state')=='Correct',25)
pick('SSRL correct but APC wrong',lambda c:c['models'].get('Spatial-SSRL',{}).get('state')=='Correct' and c['models'].get('APC-Vis',{}).get('state')=='Wrong',25)
pick('SymPL wrong but PCD correct',lambda c:c['models'].get('SymPL',{}).get('state')=='Wrong' and c['models'].get('PCD',{}).get('state')=='Correct',25)
pick('All-five-correct control',lambda c:all(c['models'].get(m,{}).get('state')=='Correct' for m in M),25)
for r in queue.values():r['proposed_split']='heldout' if int(hashlib.sha256(r['image'].encode()).hexdigest(),16)%5==0 else 'development'
write('review_queue.json',list(queue.values()))
print('review queue',len(queue),'groups',len({r['image'] for r in queue.values()}),'images')
for m in ['VST','Spatial-SSRL']:
 rr=[r for r in C if r['task']=='left_right' and m in r['models'] and r['models'][m]['state'] in ['Correct','Wrong']]
 print(m,'right',sum(r['choices'][r['models'][m]['prediction']]=='on the right' for r in rr),'/',len(rr))
for m in M:
 print(m,[(r['is_flip'],r['correct'],r['rows']) for r in flip if r['model']==m and r['task']=='left_right'])
