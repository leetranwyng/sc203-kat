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

## Chạy trên Kaggle từ GitHub public

1. Import notebook trong `notebooks/<method>/sanity.ipynb` vào Kaggle.
2. Bật **Internet**, chọn **GPU T4 ×2**, rồi Run All.
3. Notebook đã đặt `REPO_URL = 'https://github.com/leetranwyng/sc203-kat.git'` và `CODE_REF = 'v0.2.0'`. Source được clone tự động; chỉ submodule của phương pháp đang chạy được khởi tạo tại commit đã khóa.
4. Có thể gắn weights, dataset và checkpoint cũ qua Input, rồi đặt `model_input`, `dataset_input`, `resume_input`. Secret `HF_token` là tùy chọn cho Hugging Face; GitHub public không cần token.
5. Khi sanity đạt yêu cầu, dùng notebook `full.ipynb` cùng phương pháp.

Code/cache ở `/kaggle/temp`; kết quả và `source_checkout.json` ở `output_dir` dưới `/kaggle/working`. Metadata ghi commit của repo chính và submodule. Thư mục source có sửa đổi sẽ bị từ chối để tránh ghi đè code; có thể chọn cache mới qua biến môi trường `SC203_CODE_CACHE`.

Để thử code mới, đổi `CODE_REF` thành nhánh hoặc full commit đã push lên GitHub. Notebook fetch ref đó ở mỗi lần bootstrap; `main` không bị dùng lại từ local branch cũ. Dùng tag/commit cố định cho thí nghiệm cần tái lập và đổi output khi thay profile.

Tạo lại notebook từ cấu hình JSON hiện tại:

```powershell
python scripts/build_notebooks.py --ref v0.2.0
```

Generator giữ nguyên `configs/experiments/`; thay đổi tham số ngay trên Kaggle bằng cell `RUN_CONFIG`.

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

Repo chính: [leetranwyng/sc203-kat](https://github.com/leetranwyng/sc203-kat) (public). Năm submodule trỏ các fork của `leetranwyng`, khóa commit; remote `upstream` của từng submodule giữ nguồn tác giả. Xem [quản lý upstream/fork](docs/upstreams.md) và [lộ trình refactor](docs/migration.md). Giữ nguyên license/attribution trong source tác giả; xem `THIRD_PARTY_NOTICES.md`.
