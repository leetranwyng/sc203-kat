# Nguồn và lộ trình

## Đã thực hiện trong lần tách repo

- Giữ nguyên workspace cũ; copy lịch sử code sang `legacy/` (notebook đã bỏ output).
- Chuẩn bị năm upstream checkouts độc lập ở commit cố định, không dựa vào object store của repo cũ.
- Tách source runner khỏi notebook sang `profiles/`; không runtime-parse notebook cũ.
- Tạo CLI/config, notebook template, GitHub-only Kaggle bootstrap và CPU checks.
- Chuyển report mới nhất cùng figures và các summary baseline.

## Giới hạn có chủ đích

Đây là migration để bảo toàn hành vi trước, chưa phải viết lại mọi runner thành một framework mới. Các profile vẫn giữ layout output, setup và trace riêng; core chuẩn hóa phục vụ đánh giá/visualize. Adapter chạy các file Python có thứ tự trong profile; đây là các source được quản lý Git, không phải notebook chứa source nhúng.

Repo tác giả và snapshot runtime có thể khác nhau. `manifests/upstreams.json` ghi commit upstream; `profiles/<method>/manifest.json` ghi notebook/source đã di chuyển. Đặc biệt `9059ed...` của APC là hash SHA256 snapshot, không phải Git commit. Một patch tracked-worktree của SymPL được lưu để không làm mất thay đổi local.

## Bước tiếp theo khi bắt đầu cải tiến

1. Chạy sanity từng method trên Kaggle từ phiên bản GitHub này và lưu source commit trong kết quả.
2. Chuyển dần các thay đổi ổn định trong snapshot sang branch của fork nhóm, giữ regression trước/sau.
3. Tách rõ stage interface APC: detect → mask/depth → orientation → scene → transform → render → answer.
4. Thêm biến thể `apc_vis_3d` để thử reconstruction + oriented 3D boxes; giữ baseline nguyên danh tính.
5. Dùng split theo image/question group để các option permutation và ảnh flip không lọt sang tập kiểm tra hypothesis.

Đã tạo remote `leetranwyng/sc203-kat` (public) và 5 fork dưới tài khoản `leetranwyng`. Các submodule trỏ fork tương ứng và giữ remote upstream của tác giả.
