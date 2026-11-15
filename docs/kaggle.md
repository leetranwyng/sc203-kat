# Cấu hình và vận hành Kaggle

Mỗi notebook có một cell cấu hình và cùng lời gọi CLI. `method` gồm `apc_vis`, `pcd`, `sympl`, `vst`, `spatial_ssrl`. `mode` là `sanity` hoặc `full`.

- `KAT_INPUT`: đường dẫn ZIP code do `scripts/package_kaggle.py` tạo hoặc thư mục Kaggle đã giải nén. Đây là code, không phải weights.
- `REPO_URL`/`CODE_REF`: tùy chọn thay cho code Input sau khi push GitHub. Ghi rõ release hoặc commit.
- `model_input`: thư mục model HF local cho VST/SSRL/PCD; APC/SymPL có asset bundle riêng, xem `method_options` và defaults profile.
- `dataset_input`: TSV hoặc nguồn dữ liệu local mà runner tương ứng hỗ trợ. Không có thì tải bản HF đã khóa.
- `resume_input`: checkpoint/result archive từ chính method và protocol tương ứng; để `null` cho lượt mới.
- `runtime_dir`: runtime/cache tạm ở `/kaggle/temp`.
- `output_dir`: chỉ output cần giữ ở `/kaggle/working`.
- `method_options`: override tên biến gốc của profile, đã liệt kê trong `profiles/<method>/method_defaults.json` hoặc `defaults.json`.

Dùng lệnh `run --config ... --dry-run` để xem nguồn và cấu hình hiệu lực trước khi cài. Những khóa không hỗ trợ phải được báo rõ; không tự giả định chúng có tác dụng.

## Môi trường

Python kernel Kaggle chỉ chạy launcher nhẹ. Mỗi profile tạo Python bằng uv và chạy worker dưới môi trường riêng. Không cài gộp APC native dependencies vào môi trường VST. Không giả định 2 GPU là 2 worker: giữ device placement/sharding của từng profile.

## Resume và trace

Không chỉnh prompt, precision, preprocessing, model/data revision rồi dùng cùng checkpoint như một lượt chạy cũ. Thay đổi khoa học tạo experiment identity mới; logging không thay đổi câu trả lời. Worker giữ raw status để chẩn đoán dù bảng điểm gộp lỗi vào Sai.

APC/SymPL giữ các ảnh/box/mask/depth/orientation/scene/render được pipeline ghi. VST/SSRL ghi reasoning và output chứ không tạo trace tool không tồn tại. Để tránh đầy đĩa, cấu hình trace/budget trong profile, không archive weights/cache/runtime. Kết quả được lưu theo định dạng runner gốc; lệnh import/normalization dùng cho so sánh chung.

## Trạng thái kiểm chứng

Các profile được di chuyển từ source đã dùng và kiểm tra cú pháp/CPU trong lần refactor. Môi trường Linux/native và suy luận GPU cần một lượt sanity xác nhận lại sau khi chuyển repo. Việc refactor không phải một lần chạy benchmark mới.
