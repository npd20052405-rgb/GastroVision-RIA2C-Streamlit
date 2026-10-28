---
title: GastroVision AI - RIA2C
sdk: docker
app_port: 7860
---

# GastroVision AI — RIA2C M=4

Ứng dụng nghiên cứu phân loại ảnh nội soi đường tiêu hóa 22 lớp. Toàn bộ mã Python ứng dụng và bộ tự kiểm thử nằm trong **`streamlit_app.py`**. Không dùng để chẩn đoán, xác nhận tổn thương hoặc đề xuất điều trị.

## Chạy tại thư mục nghiên cứu hiện có

```powershell
python -m streamlit run streamlit_app.py
```

Mở `http://127.0.0.1:8501`. Môi trường local đã có torch 2.7.1+cu118, torchvision 0.22.1+cu118 và timm 1.0.19. Không cần cài lại các thư viện CUDA đang hoạt động.

Nếu cài môi trường mới từ mã nguồn nghiên cứu:

```powershell
python -m pip install -r deploy/requirements.txt
```

Trong gói xuất bản riêng, `requirements.txt` nằm ngay thư mục gốc. Với GPU CUDA11.8, cài torch/torchvision từ chỉ mục chính thức trước:

```text
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Trọng số và cấu hình bắt buộc

`ria2c_app_manifest.json` chứa 22 tên lớp theo thứ tự đã khóa, kiến trúc, hash20 checkpoint và10 hệ số Möbius trích từ capacity thật. Không thay hệ số để cải thiện ảnh demo. Khi ở workspace nghiên cứu, ứng dụng tìm đúng đường dẫn tương đối trong manifest.

Khi chạy ở máy khác, cấp **20 checkpoint gốc, seed42, fold0–4** vào thư mục riêng, đặt biến môi trường `RIA2C_MODEL_DIR` trỏ đến thư mục đó. Tên file phải khớp manifest:

```powershell
$env:RIA2C_MODEL_DIR = 'D:\models\ria2c'
python -m streamlit run streamlit_app.py
```

Đường dẫn trên chỉ là ví dụ; ứng dụng không hardcode ổ đĩa cá nhân. Trọng số tổng cộng3.359.593.615 byte (~3,36GB). Không đưa trọng số, dữ liệu, ảnh nội soi hoặc token lên repository mã nguồn. Thiếu file, khác dung lượng hoặc SHA-256 không khớp thì ứng dụng báo lỗi, không dự đoán mặc định. Manifest là cấu hình tin cậy; không nhận checkpoint/manifest tùy ý từ người truy cập.

## Năm khu vực sử dụng

1. **Một ảnh:** chọn JPG/JPEG/PNG và nhấn “Phân loại ảnh”. Đọc lớp, điểm Choquet, các lớp cạnh tranh và kết quả từng backbone. Chưa có nhãn thật thì không kết luận đúng/sai.
2. **Hàng loạt:** chọn22 ảnh hoặc nhiều hơn trong một lần, hoặc thêm ZIP. Xem danh sách rồi nhấn “Phân loại hàng loạt”. Tối đa128 file,20MB/ảnh,160MB/lượt,25 triệu pixel/ảnh. Ảnh hỏng báo lỗi riêng; file trùng tên có file_id riêng.
3. **Grad-CAM:** chọn bất kỳ ảnh đã xử lý, chọn target là lớp RIA2C hoặc lớp dự đoán riêng của từng backbone, rồi nhấn “Tạo Grad-CAM”. Không sử dụng heatmap có sẵn cho ảnh mới. Có thể tải PNG.
4. **Cơ chế RIA2C:** xem xác suất bốn backbone,22 điểm Choquet,10 hạng Möbius giữ dấu âm/dương. Có thể chọn bất kỳ lớp nào để giải thích và tải CSV.
5. **Nghiên cứu & hướng dẫn:** đọc class mapping, giới hạn và các khái niệm quan trọng.

Đổi tab, mở gallery hoặc chọn biểu đồ không chạy lại checkpoint. Kết quả, ảnh tải lên và CAM giữ trong bộ nhớ của phiên; không ghi ảnh người dùng ra đĩa và không tự gửi sang dịch vụ AI khác. Khi đổi ảnh/lô/target, kết quả cũ được phân biệt rõ hoặc ẩn; chỉ tính mới khi nhấn nút. Dùng “Xóa ảnh và kết quả trong phiên” khi kết thúc.

## Nhãn thật tùy chọn

CSV UTF-8:

```csv
filename,class_id
anh_01.jpg,3
anh_02.jpg,19
```

Class ID hiển thị là1–22. Không lấy nhãn từ tên ảnh. Nếu có file trùng tên, dùng `file_id,class_id` với file_id từ CSV kết quả. Chỉ tính Accuracy trên ảnh có nhãn hợp lệ và suy luận thành công; có thể cung cấp nhãn cho một phần lô. 22 ảnh không có nghĩa phải dự đoán ra22 nhãn khác nhau.

## Pipeline cố định

- XCiT-Small, ViT-B/16, MaxViT-Tiny, Swin-Tiny; mỗi mô hình trung bình xác suất softmax từ5 checkpoint.
- RGB -> Resize cạnh ngắn256 bilinear -> CenterCrop224 -> tensor -> ImageNet normalize. Không tự xoay EXIF hoặc tăng màu.
- Nạp từng checkpoint, xử lý toàn lô theo mini-batch rồi giải phóng mô hình; không giữ20 mô hình trên GPU. Một khóa tính toán ngăn nhiều phiên tranh GPU cùng lúc.
- GPU: inference_mode, FP16 autocast, softmax float32; mean5 float32 và Choquet float64. CPU: cùng mô hình, float32. Không trộn logits rồi softmax.
- RIA2C: `sum(m_i*p_i) + sum(m_ij*min(p_i,p_j))`. Capacity kiểm tra normalization và32 bất đẳng thức monotonicity. Không chuẩn hóa điểm Choquet thành xác suất.
- Grad-CAM: float32, có gradient, weights đóng băng. Cộng raw CAM có dấu của5 nhánh trước ReLU, nội suy224 và chuẩn hóa riêng. Nếu CAM suy biến thì báo rõ, không chế tạo map. Không có heatmap end-to-end RIA2C.
- Shapley capacity chỉ là thuộc tính toàn cục, không phải đóng góp theo ảnh. Möbius coefficient khác hạng đóng góp của riêng ảnh.

## Kiểm thử tái lập

Trong workspace nghiên cứu có dữ liệu/artifacts nguồn:

```text
python streamlit_app.py --self-test --report-dir reports/local_check_NEW_UNIQUE_NAME
```

Chọn thư mục chưa tồn tại. Bộ thử dùng22 ảnh thật đã có kết quả tham chiếu, thêm file hỏng; suy luận đơn/lô, đối chiếu bốn backbone và RIA2C,484 phân rã Möbius, hai chế độCAM, so sánh CAM với bốn hình nghiên cứu, kiểm tra AppTest/rerun. Đây là kiểm tra tái lập ứng dụng trên mẫu minh họa, không phải benchmark hay đánh giá nested mới. Gói public không kèm dữ liệu nên không tự chạy bộ kiểm thử nghiên cứu này.

## Docker và trạng thái cloud

Từ gói xuất bản có `Dockerfile` ở thư mục gốc:

```text
docker build -t gastrovision-ria2c .
docker run --rm --gpus all -p 7860:7860 -v /absolute/path/to/weights:/models:ro gastrovision-ria2c
```

Bỏ `--gpus all` để chạyCPU. Chưa build Docker trên máy local thì không coi image đã được kiểm thử. Hugging Face Spaces Docker cần cung cấp trọng số hợp lệ tại runtime; gói này không tự tải hoặc công khai checkpoint. Không cài token vào code hoặc Dockerfile. Dùng secrets/volume của hạ tầng đã được phê duyệt.

Chưa có website công khai chỉ nhờ đưa code lên GitHub. Chỉ tuyên bố triển khai thành công sau khi endpoint bên ngoài nạp đủ20 checkpoint và qua thử đơn/lô thật. Không tự kích hoạt GPU trả phí. Xem báo cáo bàn giao để biết trạng thái kiểm thử local, tài nguyên và phần cần cấp quyền.

## Dữ liệu, quyền sử dụng và giới hạn

Mã không kèm giấy phép tự gán cho checkpoint, backbone hoặc dataset. Cần xác nhận quyền phân phối trọng số/dữ liệu trước khi xuất bản. Ảnh mới có thể ngoài miền; hệ thống chưa có bộ phát hiện ngoài22 lớp. Xác suất cao không bảo đảm đúng, CAM nóng không xác nhận tổn thương. Không công khai ảnh chứa thông tin nhận dạng. Các kết quả nghiên cứu và hình khóa luận gốc không bị chỉnh sửa.
