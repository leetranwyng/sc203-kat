import ast,copy,json,pathlib
P=pathlib.Path(__file__).parent
n=json.loads((P/'APC_Vis_7B_complete_and_120_traces_Kaggle.ipynb').read_text(encoding='utf8'))
s=''.join(n['cells'][1]['source'])
for node in ast.walk(ast.parse(s)):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text' and "apc_prepare_assets.py" in (ast.get_source_segment(s,node.func.value) or ''):
  old=ast.literal_eval(node.args[0])
  new=old.replace('        download_checkpoint(url,target)',"        download_checkpoint('https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/c9367b194bfe0113baa1e11b2dde6a801fbfe012/sam_vit_h_4b8939.pth' if name=='sam_vit_h_4b8939.pth' else url,target)")
  assert new!=old
  compile(new,"apc_prepare_assets.py","exec")
  assert "def download_checkpoint(url,target):" in new
  s=s.replace(ast.get_source_segment(s,node.args[0]),repr(new))
  break
n['cells'][1]['source']=s.splitlines(True)
def save(nb,name):
 for c in nb['cells']:
  if c['cell_type']=='code':ast.parse(''.join(c['source']));c.update(outputs=[],execution_count=None)
 (P/name).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
save(n,'APC_Vis_7B_FULL_fixed_assets_Kaggle.ipynb')
sanity=copy.deepcopy(n)
sanity['cells']=sanity['cells'][:16]
sanity['cells'][0]['source']=['# APC-Vis 7B sanity: 3 subtasks, 6 rows\nRun T4 x2 + Internet. Same pinned model/assets as full. No benchmark resume is imported into sanity inference. Correctness is reported separately from technical execution.\n']
sanity['cells'][3]['source']+= ["\nCFG.update(output='/kaggle/working/apc_completion/sanity', experiment='apc_sanity_fixed_assets', resume_input=None, run_kind='sanity', paper_tasks=['left_right','visibility','facing'], expected_rows=6, expected_circular_groups=3, expected_task_counts=dict(left_right=2,visibility=2,facing=2), trace_keep_per_task=2,trace_max_total_mib=512,run_timeout_seconds=3600)\nOUT=Path(CFG['output']);OUT.mkdir(parents=True,exist_ok=True)\nCONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf8')\n(OUT/'config.json').write_text(json.dumps(CFG,indent=2),encoding='utf8')\n"]
sanity['cells'][12]['source']=['## Run six-row sanity and preserve all traces\n']
sanity['cells'][13]['source']=["REPORT=launch()\nprint(json.dumps(REPORT,indent=2))\nassert REPORT['attempted_rows']==6 and REPORT['valid_predictions']==6, 'Technical sanity failed: inspect saved trace and report'\nprint('Technical PASS; answer accuracy:',REPORT['row_accuracy'])\n"]
save(sanity,'APC_Vis_7B_SANITY_fixed_assets_Kaggle.ipynb')
print('Built full and sanity; syntax checked; SAM mirror keeps original provenance URL and enforced SHA256.')
