# HTTMV2

Ứng dụng nhận diện khuôn mặt chạy cục bộ trên Windows. Mỗi người dùng tự đăng ký dữ liệu trên máy của mình.

## Yêu cầu

- Windows 10/11 64-bit.
- Python 3.12 và PowerShell.
- Internet và vài GB dung lượng trống cho lần cài đầu (Python, Node.js và các model AI).
- GPU NVIDIA tương thích là tùy chọn. Hệ thống tự chọn CUDA nếu dùng được, nếu không sẽ chạy CPU.

## Cài đặt và chạy

Mở PowerShell trong thư mục muốn đặt project:

```powershell
git clone https://github.com/minhtri14012005/HTTM_BTL.git
cd HTTM_BTL
python run.py
```

Lần chạy đầu, `run.py` tự chạy `setup.ps1`. Script tạo môi trường Python riêng, cài dependencies, tải các model từ nguồn của chúng, cài frontend và dựng giao diện. Cần giữ Internet kết nối trong khi cài; quá trình tải model có thể mất thời gian. Những lần sau chỉ cần:

```powershell
python run.py
```

Nếu muốn cài trước rồi mới chạy ứng dụng:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
python run.py
```

Ứng dụng tự mở trình duyệt tại `http://127.0.0.1:8000`. Để dừng, nhấn `Ctrl+C` trong cửa sổ PowerShell hoặc nhập `Q` rồi Enter.

## Model và dữ liệu cá nhân

Các checkpoint model không được commit vào Git vì kích thước lớn và điều khoản phân phối khác nhau. `scripts/setup_models.py` tải các tệp cần thiết từ nguồn model và kiểm tra SHA-256. Máy cần truy cập các nguồn tải trong lần cài đầu.

Thư mục `data/` chứa cơ sở dữ liệu, ảnh đăng ký và dữ liệu chạy cục bộ; thư mục này bị Git bỏ qua. Không cần chia sẻ dữ liệu này để người khác cài ứng dụng. Mỗi người dùng đăng ký khuôn mặt của họ trên máy riêng. Hãy tự sao lưu `data/` nếu cần giữ dữ liệu của chính mình.

Các thư mục `.venv/`, `.tools/`, `frontend/node_modules/` và `frontend/dist/` cũng được tạo hoặc tải tại máy cục bộ, không đưa lên Git.

## Chọn thiết bị AI

Mặc định ứng dụng tự chọn CUDA hoặc CPU. Có thể ép chạy CPU trong PowerShell:

```powershell
$env:FACEWATCH_DEVICE = 'cpu'
python run.py
```

Đặt lại về tự động trong cửa sổ hiện tại:

```powershell
Remove-Item Env:FACEWATCH_DEVICE -ErrorAction SilentlyContinue
```
