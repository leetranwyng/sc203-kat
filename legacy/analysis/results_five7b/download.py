import concurrent.futures, hashlib, json, pathlib, requests, time
ROOT=pathlib.Path(__file__).parent/'raw'
SOURCES={
 'apc1':('schonfinkle/apc-part-1','apc_qwen7b/results/apc_qwen7b_full_part1_v6_fresh_5e0c9253c65f'),
 'apc2':('schonfinkle/apc-part-2','apc_qwen7b/results/apc_qwen7b_full_part2_v6_fresh_1b1945446fd7'),
 'vst':('khaitrannhu/11-11-2026','vst7b_sc203/vst7b_full_v1'),
 'ssrl':('ltrnnguyn/sc203-spatial-srrl-7b','ssrl_7b_full_paper_tasks_original'),
}
def fetch(item):
 k,name=item;ref,prefix=SOURCES[k];dest=ROOT/k/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():return dict(source=k,file=name,status='cached',bytes=dest.stat().st_size)
 url='https://www.kaggle.com/api/v1/kernels/output/download/'+ref+'/'+prefix+'/'+name
 for attempt in range(3):
  try:
   with requests.get(url,timeout=(30,180),stream=True) as r:
    if r.status_code!=200:return dict(source=k,file=name,status=r.status_code)
    if 'text/html' in r.headers.get('Content-Type',''):return dict(source=k,file=name,status='html')
    h=hashlib.sha256();tmp=dest.with_suffix(dest.suffix+'.part')
    with tmp.open('wb') as f:
     for b in r.iter_content(2**20):f.write(b);h.update(b)
    tmp.replace(dest)
   print(k,name,dest.stat().st_size,flush=True)
   return dict(source=k,file=name,status=200,bytes=dest.stat().st_size,sha256=h.hexdigest())
  except Exception as e:
   if attempt==2:return dict(source=k,file=name,status=repr(e))
   time.sleep(2)
if __name__=='__main__':
 jobs=[]
 for k in SOURCES:
  names=['checkpoint.zip'] if k=='vst' else (['results.zip','traces.zip','full_report.json','manifest.json','config.json','run/predictions_apc_vis.jsonl'] if k.startswith('apc') else ['predictions.jsonl','summary.json','diagnostics.json','effective_runtime.json','manifest.json','samples.jsonl','config.json'])
  jobs.extend((k,n) for n in names)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:out=list(ex.map(fetch,jobs))
 (ROOT/'downloads.json').write_text(json.dumps(out,indent=2),encoding='utf8')
 print(json.dumps(out,indent=2))
