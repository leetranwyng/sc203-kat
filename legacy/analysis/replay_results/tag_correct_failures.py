import collections
import html
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
p=root/'replay_report.html'
s=p.read_text(encoding='utf8')
audit=json.loads((root/'correct_case_audit.json').read_text(encoding='utf8'))
rows={r['id']:r for r in audit['apc']}
labels={
 'viewer_camera_mismatch':'APC · Viewer chọn camera',
 'viewer_other_object_mismatch':'APC · Viewer chọn vật thể khác',
 'missing_reference_detection_render_error':'APC · Thiếu detection, lỗi render',
 'color_palette_overflow':'APC · Vượt giới hạn bảng màu',
 'viewer_matches_root_cause_unresolved':'APC · Viewer khớp, chưa rõ nguyên nhân',
 'facing_viewer_review':'APC · Facing: viewer khác chủ thể, cần kiểm chứng',
 'camera_direct':'PCD · Camera sai, contrastive đúng',
 'pcd':'PCD · Contrastive sai, camera đúng',
 'required_entity_absent_from_scene':'APC · Thiếu vật thể trong scene',
 'zero_detection_candidates':'APC · Detector không tìm được vật thể',
 'refinement_default_index_zero':'APC · Chọn box mặc định do index không hợp lệ',
 'boilerplate_extracted_as_object':'APC · Trích xuất câu dẫn thành vật thể',
 'extra_objects_beyond_question':'APC · Trích xuất thừa vật thể',
 'viewer_parser_fallback_to_camera':'APC · Parser viewer rơi về camera',
 'answer_changes_meaning_between_stages':'APC · Đổi nghĩa đáp án giữa các bước',
 'translation_mentions_wrong_entity':'APC · Dịch sai tên vật thể',
 'unbalanced_quote_in_source_question':'Dữ liệu · Câu hỏi nguồn thiếu dấu ngoặc kép',
}
counts=collections.Counter();correct_flagged=0
def replace(m):
 global correct_flagged
 attrs,summary=m.groups()
 sid=html.unescape(re.search(r'data-case-id="([^"]+)"',attrs)[1])
 previous=re.search(r'data-failure="([^"]+)"',attrs)[1]
 status='correct' if previous=='correct' else 'wrong'
 errors=[]
 if status=='correct' and summary.startswith('APC · '):
  r=rows[sid]
  if 'explicit_viewer_mismatch' in r['flags']:
   errors.append('viewer_camera_mismatch' if r['evidence']['viewer']=='camera' else 'viewer_other_object_mismatch')
  errors.extend(f for f in labels if f in r['flags'])
  # Viewer mismatch for Facing remains a review observation, not a confirmed defect.
  if errors:
   correct_flagged+=1
   summary=re.sub(r'<span class="audit-tag">.*?</span>','',summary)
   summary+=''.join(f'<span class="failure-tag">{html.escape(labels[f])}</span>' for f in errors)
   summary+='<span class="audit-tag">Đáp án đúng · có bất thường trung gian</span>'
  attrs+=' data-correct-anomaly="'+('yes' if errors else 'no')+'"'
 elif status=='wrong':errors=[previous]
 counts.update(errors)
 return '<details'+attrs+f' data-outcome="{status}" data-errors="'+ ' '.join(errors)+'"><summary>'+summary+'</summary>'
s,n=re.subn(r'<details( data-failure="[^"]+"[^>]*)><summary>(.*?)</summary>',replace,s)
assert n==244,n
options='<option value="all">Tất cả · 244</option><option value="wrong">Câu sai / lỗi pipeline · 110</option><option value="correct">Câu đúng · 134</option>'
options+=f'<option value="correct_anomaly">APC đáp án đúng, có bất thường · {correct_flagged}</option>'
options+=''.join(f'<option value="{k}">{html.escape(v)} · {counts[k]}</option>' for k,v in labels.items() if counts[k])
s=re.sub(r'(<select id="failure-filter">).*?(</select>)',lambda m:m[1]+options+m[2],s,count=1)
status='<label>Trạng thái đáp án <select id="outcome-filter"><option value="all">Tất cả</option><option value="correct">Đúng</option><option value="wrong">Sai / lỗi pipeline</option></select></label>'
s=s.replace('<output id="case-count"',status+'<output id="case-count"',1)
# Remove text left by an earlier replacement that stopped at a > inside the old inline handler.
s=s.replace(' details\'))e.hidden=!e.firstChild.textContent.toLowerCase().includes(this.value.toLowerCase())">','')
s=s.replace('<main>','<p class="audit-note">Nhãn màu vàng áp dụng cho cả đáp án sai và đáp án đúng có bất thường trung gian. Một câu đúng có thể có nhiều nhãn; số lượng theo loại lỗi có thể chồng lấn. Chọn loại lỗi rồi lọc trạng thái Đúng để xem riêng các câu này. Fallback chọn box không có nghĩa box chắc chắn sai.</p><main>',1)
s=s.replace("const count = document.getElementById('case-count');", "const count = document.getElementById('case-count');\n const outcomeFilter = document.getElementById('outcome-filter');")
s=s.replace("const matches = type === 'all' || (type === 'wrong' ? tag !== 'correct' : tag === type);", "const matches = type === 'all' || (['wrong','correct'].includes(type) ? card.dataset.outcome === type : type === 'correct_anomaly' ? card.dataset.correctAnomaly === 'yes' : (card.dataset.errors || '').split(' ').includes(type));\n     const outcomeMatches = outcomeFilter.value === 'all' || card.dataset.outcome === outcomeFilter.value;")
s=s.replace('!(matches && auditMatches &&','!(matches && outcomeMatches && auditMatches &&')
s=s.replace("filter.addEventListener('change', update);", "filter.addEventListener('change', () => { auditFilter.value='all'; if(['correct','wrong','correct_anomaly'].includes(filter.value)) outcomeFilter.value='all'; update(); }); outcomeFilter.addEventListener('change', update);")
p.write_text(s,encoding='utf8')
print(json.dumps(dict(correct_apc_with_anomaly=correct_flagged,tag_counts=counts),ensure_ascii=True))
