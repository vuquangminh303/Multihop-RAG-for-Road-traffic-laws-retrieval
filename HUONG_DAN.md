# Hướng dẫn khắc phục lỗi huggingface_hub

## Lỗi phổ biến

```
ImportError: cannot import name 'hf_hub_url' from 'huggingface_hub.utils'
```

## Giải pháp đã thực hiện

Để giải quyết vấn đề này, chúng tôi đã thực hiện các thay đổi sau:

1. **Loại bỏ phụ thuộc vào `sentence-transformers`**: 
   - Đã sửa lại `retrieval.py` để sử dụng `transformers` và `torch` trực tiếp
   - Đã sửa lại `multihop_rag.py` để đảm bảo tương thích

2. **Cập nhật `requirements.txt`**:
   - Đã thêm `huggingface-hub>=0.16.4` để đảm bảo sử dụng phiên bản mới hơn
   - Đã cập nhật phiên bản `transformers`

3. **Tạo mới `update_huggingface_hub.py`**:
   - Script tự động kiểm tra và khắc phục lỗi liên quan đến huggingface_hub

## Hướng dẫn sử dụng

### Cách 1: Sử dụng script tự động cập nhật

Chạy script tự động khắc phục lỗi:

```bash
python update_huggingface_hub.py
```

Script này sẽ:
- Kiểm tra phiên bản hiện tại của `huggingface_hub` và `transformers`
- Kiểm tra xem có thể import `hf_hub_url` từ `huggingface_hub.utils` không
- Gỡ bỏ `sentence-transformers` nếu có
- Cập nhật `huggingface_hub` và `transformers` lên phiên bản mới
- Thử lại import và báo cáo kết quả

### Cách 2: Cài đặt thủ công

Nếu script tự động không hoạt động, bạn có thể thực hiện các bước sau:

1. Gỡ bỏ sentence-transformers:
   ```bash
   pip uninstall -y sentence-transformers
   ```

2. Cập nhật huggingface_hub lên phiên bản mới nhất:
   ```bash
   pip install --upgrade huggingface-hub
   ```

3. Cập nhật transformers:
   ```bash
   pip install --upgrade transformers
   ```

4. Cài đặt lại các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

### Cách 3: Sử dụng môi trường ảo mới

Nếu vẫn gặp vấn đề, bạn có thể tạo một môi trường ảo mới:

```bash
# Tạo môi trường mới với conda
conda create -n chatbot-rag python=3.10
conda activate chatbot-rag

# Hoặc với venv
python -m venv chatbot-rag
source chatbot-rag/bin/activate  # Trên Linux/Mac
chatbot-rag\Scripts\activate     # Trên Windows

# Cài đặt các thư viện
pip install -r requirements.txt
```

## Kiểm tra hệ thống

Sau khi cập nhật, bạn có thể chạy các lệnh sau để kiểm tra:

```bash
# Kiểm tra retrieval
python retrieval.py

# Chạy API server
python server.py

# Kiểm tra RAG
python multihop_rag.py
```

## Giải thích kỹ thuật

Lỗi xuất hiện do sự không tương thích giữa các phiên bản của `huggingface_hub` và `sentence-transformers`. Trong các phiên bản mới của `huggingface_hub`, hàm `hf_hub_url` đã được di chuyển hoặc đổi tên.

Giải pháp của chúng tôi là:
1. Loại bỏ phụ thuộc vào `sentence-transformers` bằng cách tự triển khai các chức năng tạo embedding
2. Cập nhật `huggingface_hub` lên phiên bản mới nhất hoặc phiên bản cụ thể có chứa `hf_hub_url`
3. Đảm bảo tương thích với cả hai phiên bản cũ và mới

## Hỗ trợ

Nếu bạn vẫn gặp vấn đề, vui lòng liên hệ với nhóm phát triển hoặc tạo issue mới trên hệ thống quản lý dự án. 