"""Display the correct-answer audit and qualify Facing viewer observations."""
import collections
import html
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
p=root/'replay_report.html'
s=p.read_text(encoding='utf8')
audit=json.loads((root/'correct_case_audit.json').read_text(encoding='utf8'))
labels={
 'explicit_viewer_mismatch':'Đúng nhưng viewer không khớp câu hỏi',
 'facing_viewer_differs_not_proven_error':'Facing: viewer khác chủ thể — chưa chứng minh lỗi',
 'target_selected_as_viewer_excluded_from_render':'Vật thể đích bị loại khỏi render do chọn làm viewer',
 'required_entity_absent_from_scene':'Thiếu vật thể bắt buộc trong scene',
 'zero_detection_candidates':'Detector không tìm được vật thể',
 'boilerplate_extracted_as_object':'Trích xuất câu dẫn thành vật thể',
 'extra_objects_beyond_question':'Trích xuất thêm vật thể ngoài câu hỏi',
 'viewer_parser_fallback_to_camera':'Parser viewer rơi về camera',
 'refinement_index_ambiguous_or_fallback':'Đầu ra chọn box không hợp lệ hoặc có nhiều chữ số',
 'refinement_default_index_zero':'Chọn box mặc định 0 do không có index hợp lệ',
 'paired_permutation_has_wrong_answer':'Lượt đảo lựa chọn cùng nhóm trả lời sai',
 'unbalanced_quote_in_source_question':'Câu hỏi nguồn có dấu ngoặc kép không đóng',
 'answer_changes_meaning_between_stages':'Câu trả lời đổi nghĩa qua các bước',
 'translation_mentions_wrong_entity':'Bước dịch nhắc sai vật thể',
 'other_branch_wrong_not_evidence_of_gating_error':'Nhánh còn lại sai — chưa chứng minh gating đúng',
}
lookup={(model.upper(),r['id']):r for model in ['apc','pcd'] for r in audit[model]}
if 'id="audit-filter"' in s:raise SystemExit('Audit already displayed.')
checked=0
def enrich(m):
 global checked
 tag,sid,summary=m.groups()
 title=html.unescape(summary.split('<span')[0]);model,task,_,_=title.split(' · ')
 if tag!='correct':
  if model=='APC' and task=='facing' and tag in ['viewer_camera_mismatch','viewer_other_object_mismatch']:
   summary=re.sub(r'<span class="failure-tag">.*?</span>', '<span class="failure-tag">APC · Facing: viewer khác chủ thể, cần kiểm chứng</span>',summary)
   return f'<details data-failure="facing_viewer_review" data-case-id="{sid}"><summary>{summary}</summary><p class="audit-correction">Đính chính: câu hỏi Facing hỏi hướng của vật thể; không bắt buộc viewer phải trùng chủ thể. Khác viewer chưa đủ chứng minh nguyên nhân sai.</p>'
  return m[0]
 r=lookup[(model,html.unescape(sid))];checked+=1
 flags=r['flags'];attr=' '.join(flags) if flags else 'no_observed_anomaly'
 chips=''.join('<span class="audit-tag">'+html.escape(labels[f])+'</span>' for f in flags if f!='refinement_index_ambiguous_or_fallback')
 if not flags:chips='<span class="audit-tag">Chưa phát hiện bất thường từ trace</span>'
 if model=='APC':
  note='Đã kiểm tra depth hữu hạn/dương, vector hướng hữu hạn/độ dài đơn vị, box hợp lệ và mask không rỗng. Các kiểm tra này không xác nhận độ chính xác hình học hay nhận diện đúng vật thể.'
 else:
  note='Trích xuất khớp câu hỏi; nhánh khớp quy tắc trong code; điểm contrastive và đáp án khớp phép tính. Chưa có ground truth hướng nhìn để xác nhận câu Yes/No của bước viewpoint.'
 detail='<details class="correct-audit"><summary>Kiểm chứng câu đúng</summary><p>'+html.escape(note)+'</p><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
 return f'<details data-failure="correct" data-audit="{attr}" data-case-id="{sid}"><summary>{summary}{chips}</summary>'+detail

s=re.sub(r'<details data-failure="([^"]+)" data-case-id="([^"]+)"><summary>(.*?)</summary>',enrich,s)
assert checked==134,checked
# Remove obsolete causal language on the 13 wrong Facing examples.
s=re.sub(r'(<p class="audit-correction">.*?</p>)<p class="failure-note">.*?</p>',r'\1',s)
s=s.replace('APC · Viewer chọn camera · 19','APC · Viewer chọn camera · 12')
s=s.replace('APC · Viewer chọn vật thể khác · 19','APC · Viewer chọn vật thể khác · 13')
s=s.replace('</select></label><output id="case-count"', '<option value="facing_viewer_review">APC · Facing: viewer khác chủ thể, cần kiểm chứng · 13</option></select></label><output id="case-count"',1)
counts=collections.Counter(f for model in ['apc','pcd'] for r in audit[model] for f in r['flags'])
opts='<option value="all">Không lọc kiểm chứng</option>'
opts+=''.join(f'<option value="{f}">{html.escape(label)} · {counts[f]}</option>' for f,label in labels.items() if counts[f])
control='<label>Kiểm chứng câu đúng <select id="audit-filter">'+opts+'</select></label>'
s=s.replace('<output id="case-count"',control+'<output id="case-count"',1)
s=s.replace('<main>','<p class="audit-note">Đã kiểm chứng trace của cả 134 lượt đúng. Nhãn kiểm chứng có thể chồng lấn. Với Facing, viewer khác chủ thể chỉ là dấu hiệu cần xem, không mặc định là lỗi. Chưa có ground truth 3D để xác nhận độ sâu/hướng chính xác.</p><main>',1)
s=s.replace('</style>','.audit-tag{display:inline-block;margin:4px;padding:4px 8px;background:#e7f0f7;color:#173e5a;border-radius:6px;font-size:14px}.correct-audit{padding:8px 12px}.audit-correction{padding:10px;background:#eef4fa;color:#173e5a}</style>',1)
s=s.replace("const count = document.getElementById('case-count');","const count = document.getElementById('case-count');\n const auditFilter = document.getElementById('audit-filter');")
s=s.replace("card.hidden = !(matches &&", "const auditMatches = auditFilter.value === 'all' || (card.dataset.audit || '').split(' ').includes(auditFilter.value);\n     card.hidden = !(matches && auditMatches &&")
s=s.replace("search.addEventListener('input', update);", "auditFilter.addEventListener('change', () => { if(auditFilter.value !== 'all') filter.value = 'correct'; update(); });\n search.addEventListener('input', update);")
p.write_text(s,encoding='utf8')
print('Annotated 134 correct cases; qualified 13 wrong Facing viewer labels.')
