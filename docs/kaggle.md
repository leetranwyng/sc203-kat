# Chạy Kaggle bằng GitHub public

## Bắt đầu

Import `notebooks/<method>/sanity.ipynb`, bật Internet và chọn GPU T4 ×2. Notebook clone `https://github.com/leetranwyng/sc203-kat.git` tại `CODE_REF` mặc định `v0.2.0`; tự khởi tạo submodule cần thiết và kiểm tra đúng commit. Không cần token GitHub hay Dataset chứa source code.

Các phương pháp: `apc_vis`, `pcd`, `sympl`, `vst`, `spatial_ssrl`. Mỗi phương pháp có sanity/full dùng cùng runner.

## Cấu hình

- `REPO_URL`: URL public của sc203-kat, đã điền sẵn.
- `CODE_REF`: tag, nhánh hoặc full commit đã có trên remote. Tag/commit cố định dùng cho kết quả tái lập; nhánh dùng khi phát triển.
- `HF_SECRET_NAME`: mặc định `HF_token`; chỉ dành cho Hugging Face. Có thể đặt `None` nếu dùng tài nguyên public/local.
- `model_input`: thư mục checkpoint local cho VST/SSRL/PCD; APC/SymPL dùng asset bundle với checkpoints và manifest riêng.
- `dataset_input`: TSV hoặc dữ liệu local theo runner; nếu trống dùng nguồn HF đã khóa.
- `resume_input`: checkpoint của đúng phương pháp/protocol, hoặc `None` cho lượt mới. ZIP checkpoint/kết quả vẫn được runner hỗ trợ.
- `runtime_dir`: môi trường và cache tạm dưới `/kaggle/temp`.
- `output_dir`: kết quả cần giữ dưới `/kaggle/working`.
- `method_options`: tham số riêng của profile; tham khảo `profiles/<method>/method_defaults.json` hoặc `defaults.json`.

Cell `RUN_CONFIG` là cấu hình dùng ngay trên Kaggle. JSON trong `configs/experiments/` được dùng khi chạy CLI hoặc sinh lại notebook. Generator giữ nguyên các JSON này.

## Luồng bootstrap

1. Clone source vào cache phân biệt theo URL/ref.
2. Fetch ref từ remote và checkout detached tại commit đã fetch; không dùng local branch cũ.
3. Khởi tạo duy nhất submodule cần dùng theo gitlink của repo chính; không dùng `submodule update --remote`.
4. Ghi config ngoài thư mục source, kiểm tra source trước khi chạy profile.
5. Runner tạo Python riêng bằng uv, cài thư viện theo phương pháp và chạy worker. Kết quả ghi commit source vào `source_checkout.json`.

Khi đổi CODE_REF, notebook lấy phiên bản đã push tương ứng. Source bẩn hoặc origin khác bị từ chối, không bị reset/ghi đè tự động. Đổi `SC203_CODE_CACHE` nếu cần một thư mục code sạch. Weights/dataset có thể gắn qua Input để giảm tải.

## Môi trường và resume

Kernel Kaggle chỉ chạy bootstrap nhẹ; worker dùng Python/thư viện riêng. Giữ cách chia hai GPU đã chọn cho từng phương pháp. Không thay prompt, preprocessing, precision hoặc model revision rồi dùng cùng checkpoint như một lượt cũ. Fail/invalid/error vẫn tính là Sai; raw status được giữ để chẩn đoán.

APC/SymPL giữ các trace hình học; VST/SSRL giữ reasoning và output. Tránh archive weights/cache/runtime vào kết quả. Các archive dùng cho checkpoint/trace/kết quả khác với source code, vốn được lấy từ GitHub.

## Kiểm chứng

Đã kiểm tra CPU và luồng clone/ref/submodule. Việc xác nhận model nạp được và inference GPU trên Kaggle vẫn cần chạy notebook sanity; đây không phải một benchmark mới.
