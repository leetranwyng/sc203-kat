# Hoàn thành VST và APC-Vis

Upload hai file `.ipynb` vào hai phiên Kaggle riêng, chọn GPU T4 ×2, bật Internet, rồi Run All. Có thể thêm secret `HF_token`. Không cần gắn lại checkpoint cũ: các notebook đã nhúng bản kết quả tải ngày 12/11/2026, có kiểm tra SHA256.

## VST

`VST_7B_RL_complete_missing_Kaggle.ipynb` giữ 1.899 lượt cũ và chạy 867 lượt còn thiếu (177 Left/Right + 690 Facing). Giữ cả Front/Behind và Visibility. Không đổi checkpoint, prompt, precision hay generation budget; không chạy baseline.

Tải `/kaggle/working/vst7b_sc203/vst7b_complete_20261112/checkpoint.zip` khi hoàn tất. Nếu vẫn thiếu, gắn checkpoint **mới nhất**, đổi `RESUME_INPUT` và tiếp tục. Giữ nguồn model HF để fingerprint trùng.

## APC

`APC_Vis_7B_complete_and_120_traces_Kaggle.ipynb` giữ 1.011 lượt part 1, chạy thêm 375 (192 LR + 183 Visibility), rồi ghép với 692 lượt part 2 đã nhúng. Không chạy lại part 2 cho benchmark.

Tiếp theo chạy lại 120 mẫu độc lập với benchmark: mỗi task LR/Visibility/Facing có 28 câu sai hợp lệ và 12 câu đúng **theo kết quả cũ**, không trùng hash ảnh trong từng task. Ưu tiên khác biệt giữa model, viewer được sửa và ba case cả năm model cùng sai. Danh sách ở `apc_trace_selection.json`. Kết quả replay có thể thay đổi; notebook ghi cả cũ/mới, không ép tỷ lệ và không ghi đè benchmark. Câu lỗi sớm chỉ có trace tới bước lỗi.

Lựa chọn còn ưu tiên đổi đáp án ngữ nghĩa khi hoán vị option và PCD có nhánh thay thế đúng. Thuật toán giảm điểm khi lặp viewer hoặc nhóm tín hiệu để tăng độ đa dạng. Trong 120 mẫu có 33 mẫu thuộc nhóm không nhất quán hoán vị, 24 mẫu viewer từng được sửa và đủ ba case cả năm cùng sai (các nhóm có thể giao nhau). Đây là heuristic ưu tiên chẩn đoán, chưa phải information gain đo được. Phiên 120 trace có thư mục và fingerprint riêng; không dùng `trace_resume.zip` của bản 60 trace để resume trực tiếp. Checkpoint benchmark part 1 vẫn dùng được như trước.

Runtime ở `/kaggle/temp`; kết quả ở `/kaggle/working/apc_completion/`:

- `combined/benchmark_results.zip`: kết quả benchmark đã ghép, đủ 2.078 dòng, báo rõ nếu vẫn còn missing.
- `part1/checkpoint.zip`: tiếp tục benchmark nếu còn thiếu; gắn bản mới nhất rồi thay `APC_RESUME_CHECKPOINT`.
- `trace_replay_120/trace_resume.zip`: trace, samples, manifest, journal; nếu chưa đủ 120, gắn ZIP này và thay `TRACE_RESUME_INPUT`. Đồng thời dùng checkpoint part 1 mới nhất để tránh chạy lại phần benchmark đã hoàn thành ở phiên trước.
- `trace_replay_120/replay_comparison.json`: số trace thực tế, kết quả cũ/mới, thay đổi đáp án và lỗi.

Trace giữ input, candidate/selected boxes, segmentation, depth, scene geometry, orientation angles/vectors, viewer audit, render và từng prompt/output/ảnh VLM đã được pipeline ghi. Nếu một công cụ lỗi thì các bước sau không tồn tại, được ghi rõ. Tổng ngân sách trace 2 GiB, mỗi mẫu 128 MiB; báo thiếu trace thay vì khẳng định đủ khi hết đĩa.

Ngân sách worker APC: 5 giờ continuation + 3 giờ replay, chưa tính setup. VST: 420 phút kể từ setup. Không bảo đảm một phiên luôn đủ; checkpoint hỗ trợ phiên tiếp theo.

Đã kiểm tra cú pháp notebook và 18 module nhúng, fingerprint resume APC, tải journal VST và từ chối fingerprint sai, số pending 375/867, lựa chọn 120 trace, gộp benchmark và mô phỏng lưu/khôi phục replay dở dang. Chưa chạy inference GPU các notebook mới trên Kaggle.
