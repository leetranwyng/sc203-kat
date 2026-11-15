"""Build two self-contained notebooks from the exact executed notebook sources."""
import ast, base64, collections, copy, hashlib, io, json, pathlib, re, zipfile
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RAW=ROOT/'results_five7b/raw'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf8'))
def lines(p):return [json.loads(x) for x in pathlib.Path(p).read_text(encoding='utf8').splitlines() if x.strip()]
def source(n,i):return ''.join(n['cells'][i]['source'])
def cell(s,kind='code'):
 c=dict(cell_type=kind,metadata={},source=s.splitlines(keepends=True))
 if kind=='code':c.update(outputs=[],execution_count=None)
 return c
def put(n,i,s):n['cells'][i]['source']=s.splitlines(keepends=True)
def write(n,name):
 for i,c in enumerate(n['cells']):
  c['id']=f'completion-{i}'
  if c['cell_type']=='code':
   s=''.join(c['source']);ast.parse(s.split('\n',1)[1] if s.startswith('%%writefile') else s)
   c.update(outputs=[],execution_count=None)
 (HERE/name).write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
 print(name)
def packed(files):
 b=io.BytesIO()
 with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
  for name,data in files.items():z.writestr(name,data)
 return b.getvalue()
def embed(blob,path):
 return f'''import base64, hashlib
from pathlib import Path
_payload = base64.b64decode({base64.b64encode(blob).decode()!r})
assert hashlib.sha256(_payload).hexdigest() == {hashlib.sha256(blob).hexdigest()!r}
_seed_path = Path({path!r}); _seed_path.parent.mkdir(parents=True, exist_ok=True)
_seed_path.write_bytes(_payload)
del _payload
'''

v=read(RAW/'vst.ipynb')
# Actual executed source has configuration and setup combined in one cell.
config_i=next(i for i,c in enumerate(v['cells']) if "RUN_NAME = 'vst7b_full_v1'" in ''.join(c['source']))
s=source(v,config_i)
s=s.replace("RUN_NAME = 'vst7b_full_v1'","RUN_NAME = 'vst7b_complete_20261112'")
s=s.replace('RESUME_INPUT = None',"RESUME_INPUT = '/kaggle/temp/vst_completion_seed.zip'",1)
s=s.replace('MODEL_INPUT = None', 'MODEL_INPUT = None',1)
put(v,config_i,embed((RAW/'vst/checkpoint.zip').read_bytes(),'/kaggle/temp/vst_completion_seed.zip')+s)
put(v,0,'''# VST-7B-RL — hoàn thành 867 lượt còn thiếu

Chọn **GPU T4 ×2**, bật **Internet**, thêm secret **HF_token** nếu cần, rồi **Run All**.
Checkpoint đã kiểm chứng ngày 12/11/2026 được nhúng trong notebook: không cần tải lại Output Kaggle hay gắn dataset checkpoint.
Giữ 1.899 kết quả cũ, chỉ chạy **177 Left/Right + 690 Object Facing**, tổng cuối **2.766**.
Front/Behind và Visibility giữ riêng; không chạy baseline, không sửa prompt/decode/model.

`RESUME_INPUT` mặc định dùng checkpoint nhúng. Nếu cần thêm phiên nữa, gắn **checkpoint.zip mới nhất** vào Input và thay đường dẫn này.
Giữ `PROTOCOL='both'`, FP16, 1.280 token và nguồn model HF để fingerprint trùng. Không đổi sang snapshot local trong lần resume này.
Output: `/kaggle/working/vst7b_sc203/vst7b_complete_20261112/`.
Notebook lưu mỗi câu và checkpoint định kỳ; hết thời gian vẫn xuất phần đã chạy. Không bảo đảm một phiên nếu tốc độ/setup thay đổi.
''')
v['cells'].append(cell('''# Xác nhận coverage cuối phiên, không coi điểm trên phần đã chạy là hoàn tất.
summary_path = OUT/'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    overall = summary['overall_selected_protocols']
    print('Saved:', overall['attempted_rows'], '/', overall['expected_rows'])
    print('Remaining:', overall['missing_rows'])
    print('COMPLETE' if overall['complete'] else 'PAUSED: use this run checkpoint.zip for the next session')
'''))
write(v,'VST_7B_RL_complete_missing_Kaggle.ipynb')

apc=read(RAW/'apc1.ipynb')
cfg=read(RAW/'apc1/config.json')
seed=packed({'manifest.json':(RAW/'apc1/manifest.json').read_bytes(),
             'run/predictions_apc_vis.jsonl':(RAW/'apc1/run/predictions_apc_vis.jsonl').read_bytes()})
frozen={}
for part in ['apc1','apc2']:
 for r in lines(RAW/part/'extracted/apc_vis_predictions.jsonl'):
  if r['status']!='missing':frozen[r['sample_id']]=r
cases=read(ROOT/'results_five7b/cases.json')
case_index={(r['task'],r['id']):r for r in cases}
chosen=[]
groups=collections.defaultdict(list)
for r in frozen.values():
 if r['status']=='ok':groups[(r['paper_task'],r['circular_group'])].append(r)
unstable={k for k,rr in groups.items() if len(rr)==2 and len({r['choices'][r['prediction']] for r in rr})>1}
for task in ['left_right','visibility','facing']:
 used=set()
 viewer_counts=collections.Counter();reason_counts=collections.Counter()
 for correct,quota in [(False,28),(True,12)]:
  pool=[r for r in frozen.values() if r['paper_task']==task and r['status']=='ok' and bool(r['correct'])==correct]
  def priority(r):
   c=case_index[task,r['sample_id']];reasons=[];score=0
   if r['sample_id'] in ['5700','5778','5832']:score+=100;reasons.append('all_five_wrong')
   states=[m['state'] for name,m in c['models'].items() if name!='APC-Vis']
   if states and all(s=='Wrong' for s in states):score+=25;reasons.append('all_available_others_wrong')
   if not correct and 'Correct' in states:score+=20;reasons.append('APC_wrong_other_model_correct')
   if correct and 'Wrong' in states:score+=15;reasons.append('APC_correct_other_model_wrong')
   if r.get('reference_viewer_strategy')=='question_subject':score+=30;reasons.append('viewer_repaired_by_question_subject')
   if (task,r['circular_group']) in unstable:score+=35;reasons.append('APC_answer_changes_under_option_permutation')
   pcd=c['models'].get('PCD',{})
   if 'alternative correct' in pcd.get('label',''):score+=12;reasons.append('PCD_alternative_branch_correct')
   return score,reasons
  def viewer(r):
   m=re.search(r"stand at the (.+?)'s position",r['question'],re.I)
   return (m.group(1) if m else r.get('resolved_viewer') or r['question']).casefold()
  def utility(r):
   score,reasons=priority(r)
   return score-6*viewer_counts[viewer(r)]-2*sum(reason_counts[x] for x in reasons),-int(r['sample_id'])
  selected=[]
  while len(selected)<quota:
   pool=[r for r in pool if r['image_sha256'] not in used]
   if not pool:break
   r=max(pool,key=utility)
   h=r['image_sha256']
   used.add(h);selected.append(dict(sample_id=r['sample_id'],paper_task=task,image_sha256=h,
     prior_prediction=r['prediction'],prior_correct=correct,prior_status=r['status'],
     information_score=priority(r)[0],selection_utility=utility(r)[0],viewer_text=viewer(r),
     selection_reasons=priority(r)[1] or ['diverse_image_control' if correct else 'diverse_image_failure']))
   viewer_counts[viewer(r)]+=1
   reason_counts.update(priority(r)[1])
  assert len(selected)==quota
  chosen.extend(selected)
(HERE/'apc_trace_selection.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2),encoding='utf8')
frozenblob=packed({'part2_samples.jsonl':(RAW/'apc2/extracted/samples.jsonl').read_bytes(),
 'part2_predictions.jsonl':(RAW/'apc2/extracted/apc_vis_predictions.jsonl').read_bytes(),
 'trace_selection.json':json.dumps(chosen).encode()})
boot=source(apc,1)
boot=boot.replace("WORK=Path('/kaggle/working/apc_qwen7b')", "WORK=Path('/kaggle/temp/apc_completion_runtime')")
boot=boot.replace('from apc_full_checkpoint import export_full','from apc_full_checkpoint import prepare_resume,export_full')
boot+='\n'+embed(seed,'/kaggle/temp/apc_completion_seed.zip')+embed(frozenblob,'/kaggle/temp/apc_completion_reference.zip')
boot+='\n'+f"(SCRIPTS/'completion_helpers.py').write_text({(HERE/'completion_helpers.py').read_text(encoding='utf8')!r},encoding='utf8')\n"
put(apc,1,boot)
# Keep every inference identity field unchanged; only operational settings differ.
cfg.update(experiment='apc_part1_complete_20261112',work='/kaggle/temp/apc_completion_runtime',
 output='/kaggle/working/apc_completion/part1',repo='/kaggle/temp/apc_completion_runtime/repo_9059ed359baf',
 resume_input='/kaggle/temp/apc_completion_seed.zip',trace_keep_per_task=0,trace_max_total_mib=128,
 run_timeout_seconds=18000)
put(apc,3,'''# Optional: point this to a NEWER part-1 checkpoint if continuing another session.
APC_RESUME_CHECKPOINT = '/kaggle/temp/apc_completion_seed.zip'
# Trace resume ZIP from this notebook; None starts the selected replay from scratch.
TRACE_RESUME_INPUT = None
RUN_TRACE_REPLAY = True
'''+f'CFG = {cfg!r}\n'+'''CFG.update(work=str(WORK), repo=str(REPO), resume_input=APC_RESUME_CHECKPOINT)
OUT=Path(CFG['output']); OUT.mkdir(parents=True,exist_ok=True)
CONFIG=WORK/'completion_config.json'
with setup_stage('resume_configuration',OUT,CFG):
    prepare_resume(CFG,WORK)
CONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf8')
(OUT/'config.json').write_text(json.dumps(CFG,indent=2),encoding='utf8')
print('Part 1: retain 1011 rows; original pending count 375. No retries of completed rows.')
''')
# Source's early stop is only a fresh-run check, not suitable for resuming.
put(apc,11,'''from apc_launcher import launch as launch_full
def launch():
    return launch_full(CFG,PY,SCRIPTS,ENV,retry_failed=False)
''')
put(apc,0,'''# APC-Vis 7B — hoàn thành 375 lượt + 120 trace chọn lọc

Chọn **T4 ×2**, bật **Internet**, có thể thêm secret **HF_token**, rồi **Run All**.
Notebook đã nhúng checkpoint **1.011 lượt part 1** và kết quả **692 lượt part 2** tải ngày 12/11/2026.
Không cần gắn lại kết quả cũ. Chỉ chạy thêm **192 Left/Right + 183 Visibility**, không chạy lại Facing cho benchmark.

Sau đó replay **40 ảnh mỗi task**: Left/Right, Visibility, Object Facing; mỗi task **28 câu sai hợp lệ + 12 câu đúng theo lần chạy trước**.
Ưu tiên ba case cả năm model sai, APC bất đồng với model khác, đổi đáp án khi hoán vị lựa chọn, viewer từng được sửa và PCD chọn sai nhánh. Chọn tham lam có phạt lặp viewer/nhóm tín hiệu để đa dạng mẫu; mỗi task không trùng hash ảnh. Đây là điểm ưu tiên chẩn đoán từ kết quả đã có, không phải information gain đã đo thực nghiệm.
Replay ghi cả kết quả cũ/mới và thay đổi đáp án. Vì vậy số đúng/sai của lần replay có thể khác 28/12; không ép đáp án để đạt quota.
Ground truth chỉ dùng chọn mẫu chẩn đoán, không được đưa thêm vào prompt inference. Đây là tập chọn có chủ đích, không dùng accuracy của 120 câu để đại diện benchmark.

Benchmark và replay lưu riêng. Lỗi viewer vẫn được lưu; case lỗi sớm chỉ có trace tới bước hỏng, không giả tạo render.
Runtime/weights ở `/kaggle/temp`, kết quả ở `/kaggle/working/apc_completion/`.
Thời gian worker: tối đa 5 giờ cho phần còn thiếu, 3 giờ replay (không tính setup); mỗi phần có checkpoint để tiếp tục nếu bị ngắt.
''')
put(apc,2,'## Resume cấu hình đã chạy thành công\nGiữ model, prompt, precision, parser, viewer policy và seed. Chỉ đổi đường dẫn runtime/output và ngân sách lưu trace.')
put(apc,10,'## Tiếp tục benchmark\nWorker bỏ qua mọi sample ID đã lưu và chỉ xử lý phần còn thiếu.')
put(apc,12,'## Chạy các lượt còn thiếu')
apc['cells'].extend([
 cell('## Gộp benchmark, sau đó replay 120 mẫu\nKết quả replay không ghi đè đáp án benchmark. Nếu replay bị ngắt, lấy `trace_resume.zip` của phiên này, đặt `TRACE_RESUME_INPUT` và chạy lại notebook.','markdown'),
 cell('''from completion_helpers import merge_benchmark, prepare_replay, finish_replay
REFERENCE=Path('/kaggle/temp/apc_completion_reference.zip')
merge_benchmark(OUT,REFERENCE,Path('/kaggle/working/apc_completion/combined'))
if RUN_TRACE_REPLAY:
    TRACE_CFG=prepare_replay(CFG,OUT,REFERENCE,WORK,SCRIPTS,PY,DATA_ENV,TRACE_RESUME_INPUT)
    try:
        TRACE_REPORT=launch_full(TRACE_CFG,PY,SCRIPTS,ENV,retry_failed=False)
    finally:
        finish_replay(Path(TRACE_CFG['output']),REFERENCE)
'''),
 cell('''## Tải kết quả
- `combined/benchmark_results.zip`: toàn bộ 2.078 dòng, giữ cả missing/error nếu chưa hoàn tất.
- `part1/checkpoint.zip`: dùng tiếp nếu benchmark vẫn còn thiếu.
- `trace_replay_120/trace_resume.zip`: metadata + predictions + trace; dùng tiếp khi replay bị ngắt.
- `trace_replay_120/replay_comparison.json`: đủ/thiếu trace, đáp án cũ/mới và số đúng/sai mới theo task.
- `trace_replay_120/traces.zip`: trace do exporter tạo nếu đủ ngân sách đĩa; `trace_resume.zip` cũng chứa trace đã giữ.

Không tải runtime/weights. Notebook không tự xóa dữ liệu benchmark để dành đĩa. Nếu gặp giới hạn tài nguyên, xem report và resume từ checkpoint mới nhất.
''','markdown')])
write(apc,'APC_Vis_7B_complete_and_120_traces_Kaggle.ipynb')
