# Kế hoạch reproduce VST-7B-RL trên Kaggle

## Checkpoint và phạm vi

Repo cục bộ VST được đọc tại commit `e373b3d498d8d5e4b0160ad40542fed98079f683`. Paper `E:/SC203/VST/paper.pdf`, Bảng 2 (trang 8), báo **60.1** trên 3DSRBench cho VST-7B-RL. Repo có hướng dẫn inference Transformers trong README và cấu hình đánh giá 3DSRBench trong `benchmark/config/config_vlm3dreasoner.json`. Hai notebook tải checkpoint `rayruiyang/VST-7B-RL`, revision `b18ec54b43e1d0b8a171cf41d9f221fbedd803ae`, với 4 shard safetensors (~16,6 GB). Đây là đánh giá checkpoint đã SFT/RL trên tập con SC203. Điểm của paper dùng làm tham chiếu cho kết quả công bố.

Dữ liệu `ccvl/3DSRBench` revision `5c90e422b68d0484d50a3ff06b3096ea7d288026`. Giữ ảnh, QID và thứ tự A/B của TSV CircularEval.

| Tác vụ | Lượt | Nhóm CircularEval | Đối chiếu |
|---|---:|---:|---|
| Left/Right | 698 | 349 | APC-Vis, PCD, SymPL |
| Object Facing | 692 | 346 | APC-Vis, SymPL |
| Front/Behind gốc | 688 | 344 | PCD |
| Visibility yes/no từ Front/Behind | 688 | 344 | APC-Vis, SymPL |

`both` chạy 2.766 lượt. Dùng `original` hoặc `visibility` để giảm còn 2.078 lượt. Một nhóm CircularEval đúng khi cả hai hoán vị đều đúng. Hai dạng Front/Behind và Visibility có điểm riêng. Báo cáo dùng exact-choice, với parser chỉ đọc phần cuối sau `</think>` nếu có.

## Trình tự chạy

1. **Sanity.** Trên Kaggle bật Internet, chọn T4 ×2, rồi Run All notebook sanity. Ảnh máy bay–xe đạp, ID 5998 và 5999, kiểm tra việc xử lý một ảnh với hai hoán vị. Kiểm tra `runtime.json` có device map qua hai GPU và `summary.json` có 2/2 lượt, không có invalid. Kiểm tra đạt kỹ thuật độc lập với đúng/sai của câu hỏi.
2. **Full.** Giữ chế độ FP16, SDPA và greedy với ngân sách mặc định tối đa 1.280 token sinh. Notebook thử một cặp thật mỗi tác vụ trước và lưu các lượt pilot vào kết quả. Đọc tốc độ đo được sau 25–100 lượt. Thời gian suy luận phụ thuộc độ dài đầu ra và có thể vượt thời lượng một phiên.
3. **Chia phiên nếu cần.** Dùng `MAX_ROWS_THIS_SESSION` hoặc `SESSION_BUDGET_MINUTES`. Tải `checkpoint.zip` sau phiên. Phiên mới gắn ZIP qua Kaggle Input và đặt `RESUME_INPUT`. Giữ cùng model, dữ liệu, prompt, giới hạn token và chế độ. Dùng `sample_key` để ghép khi chạy cả hai giao thức.
4. **Phân tích.** Lấy `summary.json` để báo cáo accuracy và CircularEval theo task. `predictions.jsonl` giữ câu hỏi, lựa chọn, hash ảnh, prompt, raw output, final text, lý do parse, thời gian và số token. `runtime.json` lưu cách chia weight và peak VRAM trên mỗi GPU. Kết quả tạm có số lượt thiếu và trạng thái complete.

## Các lỗi được xử lý trong thiết kế

- **Python và CUDA.** Dùng uv để tạo môi trường Python 3.11 và PyTorch cu124. Chạy phép tính FP16 nhỏ trên từng GPU trước khi tải checkpoint. Dùng Transformers/Accelerate và SDPA để tránh đòi FlashAttention 2 trên T4.
- **VRAM.** Chia một bản model qua hai T4. Mỗi GPU có giới hạn cấp weight 12 GiB. Worker xác nhận device map dùng cả hai GPU. Batch size 1 và ảnh giới hạn số visual token.
- **Hết ổ đĩa.** Runtime, checkpoint model, TSV và ảnh dedup nằm ở `/kaggle/temp` hoặc Input. `/kaggle/working` chỉ nhận kết quả gọn. Kiểm tra dung lượng trước các bước lớn. ZIP checkpoint ghi tệp tạm rồi thay thế bản cũ. Có thể trỏ `MODEL_INPUT` vào snapshot chính thức đã gắn để tránh tải lại.
- **Runtime hoặc parser lỗi.** Lưu từng câu ngay sau generation. Lỗi worker lưu traceback và checkpoint phần đã chạy. Đầu ra bị cắt, chưa đóng think hoặc không khớp lựa chọn nhận lý do riêng. Mẫu số đầy đủ và mẫu số phần đã chạy được báo cáo riêng.
- **Resume và provenance.** Fingerprint khóa model, dữ liệu, sample, prompt, parser, decode và phiên bản thư viện. Chỉ sửa phần đuôi JSONL dở dang khi resume. Trọng số gắn cục bộ được hash và ghi là nguồn local. Giữ cùng nguồn trọng số giữa các phiên để fingerprint khớp.

Nguồn chính thức: [paper](https://arxiv.org/abs/2511.05491), [repo](https://github.com/Yangr116/VST), [checkpoint](https://huggingface.co/rayruiyang/VST-7B-RL), [dataset](https://huggingface.co/datasets/ccvl/3DSRBench).
