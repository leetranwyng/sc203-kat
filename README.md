# sc203-kat

Code và thí nghiệm SC203 về object-centric spatial reasoning trên 3DSRBench, với APC-Vis, PCD, SymPL, VST-7B-RL và Spatial-SSRL-7B.

Repo này được tách từ workspace đã hoàn thành baseline. `profiles/` giữ source chạy và các sửa lỗi đã dùng, `third_party/` giữ repo tác giả theo commit, còn `notebooks/` chỉ chứa cấu hình và lời gọi runner. Các notebook lịch sử nằm ở `legacy/` và không được đọc để chạy profile mới.

## Bắt đầu

```powershell
cd E:/sc203-kat
python -m pip install -e .
python -m sc203_kat run --config configs/experiments/vst_sanity.json --dry-run
python -m unittest discover -s tests -v
```

`--dry-run` kiểm tra cấu hình và source, không tải model hoặc yêu cầu GPU. Chạy inference thực tế dùng Kaggle Linux và môi trường của từng phương pháp.

## Đưa lên Kaggle ngay, chưa cần GitHub

```powershell
python scripts/package_kaggle.py --method apc_vis --output dist/sc203-kat-apc-vis.zip
```

1. Upload gói code ZIP lên một Kaggle Dataset, gắn làm Input. Nếu Kaggle giải nén ZIP, dùng đường dẫn thư mục đã giải nén.
2. Import `notebooks/apc_vis/sanity.ipynb`, chọn GPU T4 ×2 và bật Internet để cài thư viện/tải checkpoint chưa gắn sẵn.
3. Trong cell đầu, đặt `KAT_INPUT` trỏ tới gói code hoặc thư mục Input; điền `model_input`, `dataset_input`, `resume_input` nếu đã có các asset đó. Đọc [hướng dẫn cấu hình](docs/kaggle.md) cho đường dẫn asset riêng của APC/SymPL.
4. Có thể thêm Secret `HF_token` hoặc đổi `HF_SECRET_NAME` trong cell cấu hình. Token chỉ được truyền qua môi trường, không lưu vào config/kết quả.
5. Chạy sanity trước, sau đó dùng `full.ipynb`. Cả hai dùng cùng source profile. Bản refactor đã kiểm tra CPU; cần xác nhận sanity GPU sau khi di chuyển code.

Notebook cũng hỗ trợ `REPO_URL` và `CODE_REF` (release/commit cụ thể). Không dùng code trôi theo `main` cho kết quả cần tái lập. Repo chính hiện private: trên Kaggle nên dùng gói code Input; chế độ clone cần đăng nhập GitHub có quyền đọc repo.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/sc203_kat/` | CLI, dữ liệu, scoring, journal, trace, phân tích |
| `profiles/<method>/` | Source và các bước môi trường/inference riêng từng phương pháp |
| `third_party/` | 5 Git submodules; commit và nguồn trong `manifests/upstreams.json` |
| `configs/` | Task/protocol và cấu hình sanity/full |
| `environments/` | Ghi nhận môi trường riêng từng profile |
| `notebooks/` | 10 notebook mỏng, sinh từ một template |
| `experiments/baselines_7b/` | Bảng kết quả, metadata baseline đã chạy |
| `reports/week2/` | Report LaTeX, PDF và toàn bộ ảnh trong `figures/` |
| `legacy/` | Notebook, generator, script cũ để đối chiếu |
| `tests/` | Kiểm tra CPU; không giả lập xác nhận chất lượng GPU |

## Kết quả và scoring

Visibility và Front/Behind là hai protocol riêng. Fail/invalid/error tính là Sai; trạng thái gốc vẫn được lưu. Benchmark và replay trace chọn lọc không ghi đè nhau. Các summary baseline được lưu trong `experiments/baselines_7b/`; weights, toàn bộ ảnh, runtime và archive trace lớn vẫn ở kho dữ liệu ngoài repo.

```powershell
python -m sc203_kat import-results --source E:/SC203/APC-VLM/results_five7b --output artifacts/baselines_7b
```

CLI in ra đường dẫn prediction chuẩn hóa để dùng cho `evaluate` và `visualize`. Các trace gốc vẫn được liên kết tới workspace cũ; muốn mang sang máy khác cần chuyển artifact tương ứng.

## Git và nguồn gốc

Repo chính: [leetranwyng/sc203-kat](https://github.com/leetranwyng/sc203-kat) (private). Năm submodule trỏ các fork của `leetranwyng`, khóa commit; remote `upstream` của từng submodule giữ nguồn tác giả. Xem [quản lý upstream/fork](docs/upstreams.md) và [lộ trình refactor](docs/migration.md). Giữ nguyên license/attribution trong source tác giả; xem `THIRD_PARTY_NOTICES.md`.
