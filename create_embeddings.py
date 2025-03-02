#!/usr/bin/env python3
import json
import os
import sys
import numpy as np
import faiss
import time

print("Bắt đầu tạo embeddings (phiên bản độc lập)...")

# Kiểm tra file metadata.json đã tồn tại chưa
if not os.path.exists("metadata.json"):
    print("Lỗi: File metadata.json không tồn tại. Hãy chạy create_metadata.py trước.")
    sys.exit(1)

try:
    print("Đang nhập các thư viện cần thiết...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    print("✓ Đã nhập các thư viện thành công!")
    
    # Hàm tạo embeddings trung bình từ hidden states (mean pooling)
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Đọc dữ liệu metadata
    print("Đang đọc metadata từ file...")
    with open("metadata.json", "r", encoding="utf-8") as f:
        segment_data = json.load(f)
    
    # Lấy nội dung các đoạn
    contents = [seg["content"] for seg in segment_data]
    print(f"Đã tìm thấy {len(contents)} đoạn văn bản để tạo embeddings.")
    
    # Tải model multilingual từ huggingface
    print("Đang tải model đa ngôn ngữ...")
    # Sử dụng mô hình tương đương với distiluse-base-multilingual-cased-v2
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Đặt model ở chế độ evaluation
    model.eval()
    print("✓ Đã tải model thành công!")
    
    # Tạo embeddings
    print("Đang tạo vector embeddings...")
    start_time = time.time()
    batch_size = 16  # Batch size nhỏ hơn để tiết kiệm bộ nhớ
    embeddings = []
    total_batches = (len(contents) + batch_size - 1) // batch_size
    
    # Xử lý theo batch
    for i in range(0, len(contents), batch_size):
        batch = contents[i:i+batch_size]
        batch_end = min(i+batch_size, len(contents))
        print(f"Xử lý batch {i//batch_size + 1}/{total_batches} (đoạn {i+1}-{batch_end}/{len(contents)})...", end="", flush=True)
        
        # Tokenize và tính toán embeddings
        encoded_input = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean Pooling 
        batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Chuyển đổi thành numpy array
        batch_embeddings = batch_embeddings.detach().numpy()
        embeddings.append(batch_embeddings)
        print(" ✓")
    
    # Gộp tất cả batch embeddings
    embeddings = np.vstack(embeddings)
    print(f"✓ Đã tạo embeddings với shape: {embeddings.shape}")
    end_time = time.time()
    print(f"Thời gian tạo embeddings: {end_time - start_time:.2f} giây")
    
    # Tạo FAISS index
    print("Đang tạo FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Lưu FAISS index
    print("Đang lưu FAISS index...")
    faiss.write_index(index, "faiss_index.bin")
    
    # Lưu danh sách segment_ids
    print("Đang lưu danh sách segment_ids...")
    segment_ids = [seg["segment_id"] for seg in segment_data]
    with open("segment_ids.json", "w", encoding="utf-8") as f:
        json.dump(segment_ids, f, ensure_ascii=False)
    
    print("\n===== HOÀN THÀNH =====")
    print(f"✓ Đã lưu {len(segment_ids)} segment_ids vào segment_ids.json")
    print(f"✓ Đã tạo FAISS index với {index.ntotal} vectors và lưu vào faiss_index.bin")
    print("\nBạn có thể sử dụng retrieval.py để kiểm tra kết quả!")

except ImportError as e:
    print(f"\nLỗi khi import thư viện: {e}")
    print("\nGợi ý khắc phục lỗi import:")
    print("1. Cài đặt các thư viện cần thiết:")
    print("   pip install transformers torch faiss-cpu")
    sys.exit(1)
except Exception as e:
    print(f"\nLỗi khi tạo embeddings: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 