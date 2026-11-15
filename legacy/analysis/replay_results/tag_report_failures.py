"""Add audited, non-causal failure labels to the existing self-contained report."""
import collections
import html
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
path = root / 'replay_report.html'
source = path.read_text(encoding='utf-8')
audit = json.loads((root / 'failure_case_counts.json').read_text(encoding='utf-8'))
labels = {
    'facing_viewer_review': 'APC · Facing: viewer khác chủ thể, cần kiểm chứng',
    'viewer_camera_mismatch': 'APC · Viewer chọn camera',
    'viewer_other_object_mismatch': 'APC · Viewer chọn vật thể khác',
    'missing_reference_detection_render_error': 'APC · Thiếu detection, lỗi render',
    'color_palette_overflow': 'APC · Vượt giới hạn bảng màu',
    'viewer_matches_root_cause_unresolved': 'APC · Viewer khớp, chưa rõ nguyên nhân',
    'camera_direct': 'PCD · Camera sai, contrastive đúng',
    'pcd': 'PCD · Contrastive sai, camera đúng',
}
lookup = {(model, r['id']): r for model, rows in [('APC', audit['apc_rows']), ('PCD', audit['pcd_rows'])] for r in rows}
counts = collections.Counter()
seen = set()

def replace(match):
    title = html.unescape(match[1])
    if not title.startswith(('APC · ', 'PCD · ')):
        return match[0]
    model, task, sid, status = title.split(' · ')
    row = lookup[(model, sid)]
    seen.add((model, sid))
    tag = (row['bucket'] if model == 'APC' else row['branch']) if row['wrong'] else 'correct'
    counts[tag] += 1
    opening = f'<details data-failure="{tag}" data-case-id="{html.escape(sid, quote=True)}"><summary>{match[1]}'
    if not row['wrong']:
        return opening + '</summary>'
    badge = f'<span class="failure-tag">{labels[tag]}</span>'
    if model == 'APC':
        description = f"Vật thể tham chiếu theo câu hỏi: {row['expected_viewer']}. Viewer trong trace: {row['actual_viewer']}. "
        if tag == 'facing_viewer_review':
            description += 'Facing không bắt buộc viewer phải trùng chủ thể; chưa thể gán đây là nguyên nhân sai.'
        elif tag.startswith('viewer_') and row['viewer_mismatch']:
            description += 'Quan sát được viewer không khớp; chưa kiểm chứng rằng đổi viewer sẽ sửa được đáp án. '
            description += 'Vật thể tham chiếu có trong scene.' if row['reference_present_in_scene'] else 'Vật thể tham chiếu không có trong scene; cần kiểm tra extraction/detection.'
        elif tag == 'viewer_matches_root_cause_unresolved':
            description += 'Chưa gán lỗi cho orientation, depth, render hay suy luận nếu chưa có kiểm chứng riêng.'
        elif tag == 'color_palette_overflow':
            description += 'Trace báo IndexError tại color_dict[obj_idx].'
        else:
            description += 'Detector không tìm thấy monkey; renderer không còn vật thể sau khi bỏ viewer.'
    else:
        description = 'Nhánh được chọn trả lời sai, nhánh còn lại có đáp án đúng trong replay. Đây là quan sát bất đồng giữa hai nhánh, chưa chứng minh bộ chọn nhánh là nguyên nhân duy nhất.'
    return opening + badge + '</summary><p class="failure-note">' + html.escape(description) + '</p>'

if 'id="failure-controls"' in source:
    raise SystemExit('Report already tagged; avoid duplicate annotations.')
source = re.sub(r'<details><summary>(.*?)</summary>', replace, source)
assert len(seen) == 244 and counts['correct'] == 134, (len(seen), counts)
assert sum(counts.values()) - counts['correct'] == 110
options = '<option value="all">Tất cả · 244</option><option value="wrong">Chỉ câu sai / lỗi pipeline · 110</option><option value="correct">Câu đúng · 134</option>'
options += ''.join(f'<option value="{k}">{v} · {counts[k]}</option>' for k, v in labels.items())
controls = '<div id="failure-controls"><label>Tìm câu <input id="case-search" placeholder="ID, subtask, phương pháp hoặc nhãn lỗi"></label><label>Loại lỗi <select id="failure-filter">' + options + '</select></label><output id="case-count" aria-live="polite"></output></div><p class="audit-note">Nhãn dựa trên trace của tập replay đã chọn. 71 lượt APC và 39 lượt PCD không đúng, có tính lượt đảo lựa chọn. Mỗi lượt sai có một nhãn chính, không tính trùng lỗi dây chuyền. Visibility của APC và Front/Behind của PCD được giữ riêng.</p>'
source, n = re.subn(r'<input placeholder=.*?>', lambda _: controls, source, count=1)
assert n == 1
source = source.replace('</style>', '''
body{padding:0 16px} .failure-tag{display:inline-block;margin:4px 0 4px 12px;padding:4px 8px;border-radius:6px;background:#fff0d6;color:#663b00;font-size:14px}
.failure-note{padding:10px 12px;background:#fff8eb;border-left:3px solid #b7761b;line-height:1.6}
#failure-controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end}#failure-controls label{display:flex;flex:1 1 260px;flex-direction:column;gap:6px}
#failure-controls input,#failure-controls select{width:100%;box-sizing:border-box;padding:10px;font:inherit}#case-count{flex-basis:100%}
summary{overflow-wrap:anywhere}.audit-note{line-height:1.6}main>details{border-bottom:1px solid #ddd}pre{overflow-wrap:anywhere}
</style>''', 1)
source += '''<script>
(() => {
 const search = document.getElementById('case-search');
 const filter = document.getElementById('failure-filter');
 const count = document.getElementById('case-count');
 const cards = [...document.querySelectorAll('main > details')];
 function update(){
   const q = search.value.toLocaleLowerCase().trim(), type = filter.value;
   let shown = 0;
   for(const card of cards){
     const tag = card.dataset.failure;
     const matches = type === 'all' || (type === 'wrong' ? tag !== 'correct' : tag === type);
     card.hidden = !(matches && card.querySelector('summary').textContent.toLocaleLowerCase().includes(q));
     if(!card.hidden) shown++;
   }
   count.textContent = `Hiển thị ${shown} / ${cards.length} lượt replay`;
 }
 search.addEventListener('input', update); filter.addEventListener('change', update); update();
})();
</script>'''
path.write_text(source, encoding='utf-8')
print(json.dumps(dict(counts), ensure_ascii=True))
