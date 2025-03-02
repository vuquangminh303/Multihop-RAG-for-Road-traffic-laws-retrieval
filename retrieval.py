import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

# Hàm mean pooling như trong create_embeddings_fixed.py
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class VectorRetriever:
    def __init__(self, model_name="sentence-transformers/distiluse-base-multilingual-cased-v2"):
        # Sử dụng transformers trực tiếp thay vì sentence_transformers
        print(f"Đang tải model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Chế độ evaluation
        
        try:
            # Load dữ liệu và index
            with open("metadata.json", "r", encoding="utf-8") as f:
                self.segment_data = json.load(f)
                
            with open("segment_ids.json", "r", encoding="utf-8") as f:
                self.segment_ids = json.load(f)
                
            # Tạo mapping từ segment_id đến index trong list và ngược lại
            self.segment_id_to_index = {seg_id: i for i, seg_id in enumerate(self.segment_ids)}
            
            # Load FAISS index
            self.index = faiss.read_index("faiss_index.bin")
            print(f"Đã tải dữ liệu thành công: {len(self.segment_data)} đoạn văn bản, {self.index.ntotal} vector")
        except Exception as e:
            print(f"Lỗi khi tải dữ liệu: {e}")
            raise
    
    # Hàm encode tương tự như trong create_embeddings_fixed.py
    def encode(self, texts):
        # Đảm bảo texts là list
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize và tính toán embeddings
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Mean Pooling
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.detach().numpy()
    
    # Hàm lấy các đoạn tương tự với một đoạn cho trước
    def get_similar_segments(self, segment_id, m=5):
        if segment_id not in self.segment_id_to_index:
            return []
        
        segment_index = self.segment_id_to_index[segment_id]
        # Lấy embedding từ index
        segment_vector = np.zeros((1, self.index.d), dtype=np.float32)
        self.index.reconstruct(segment_index, segment_vector[0])
        
        # Tìm kiếm đoạn tương tự
        D, I = self.index.search(segment_vector, m+1)
        similar_indices = I[0][1:]  # Bỏ qua chính nó
        
        # Lấy thông tin đoạn
        similar_segments = [self.segment_data[idx] for idx in similar_indices if idx < len(self.segment_data)]
        return similar_segments

    # Hàm multihop retrieval
    def multihop_retrieve(self, query, k=5, m=3):
        # Bước 1: Truy xuất top-k đoạn dựa trên truy vấn
        query_embedding = self.encode([query])
        D, I = self.index.search(query_embedding, k)
        first_hop_indices = I[0]
        first_hop_segments = [self.segment_data[i] for i in first_hop_indices if i < len(self.segment_data)]
        
        # Bước 2: Từ mỗi đoạn trong top-k, truy xuất top-m đoạn tương tự
        second_hop_segments = []
        for segment in first_hop_segments:
            segment_id = segment["segment_id"]
            similar_segments = self.get_similar_segments(segment_id, m)
            second_hop_segments.extend(similar_segments)
        
        # Kết hợp và loại bỏ trùng lặp
        all_segments = first_hop_segments + second_hop_segments
        unique_segments = {}
        for seg in all_segments:
            if seg["segment_id"] not in unique_segments:
                unique_segments[seg["segment_id"]] = seg
        
        # Bước 3: Xếp hạng lại dựa trên độ tương đồng với truy vấn gốc
        ranked_segments = []
        for segment_id, segment in unique_segments.items():
            if segment_id in self.segment_id_to_index:
                idx = self.segment_id_to_index[segment_id]
                segment_vector = np.zeros((1, self.index.d), dtype=np.float32)
                self.index.reconstruct(idx, segment_vector[0])
                distance = np.linalg.norm(segment_vector - query_embedding)
                ranked_segments.append((segment, distance))
        
        # Sắp xếp theo khoảng cách tăng dần
        ranked_segments.sort(key=lambda x: x[1])
        return [seg for seg, _ in ranked_segments]
    
    def simple_retrieve(self, query, k=5):
        """Hàm truy xuất đơn giản không dùng multihop"""
        query_embedding = self.encode([query])
        D, I = self.index.search(query_embedding, k)
        results = [self.segment_data[i] for i in I[0] if i < len(self.segment_data)]
        return results

# Ví dụ sử dụng
if __name__ == "__main__":
    print("Khởi tạo Vector Retriever...")
    try:
        retriever = VectorRetriever()
        
        print("\nNhập câu hỏi để tìm kiếm thông tin (nhập 'exit' để thoát):")
        while True:
            query = input("\nCâu hỏi: ")
            if query.lower() == 'exit':
                break
                
            print("\nĐang tìm kiếm thông tin...")
            
            print("\nKết quả truy xuất đơn giản:")
            results = retriever.simple_retrieve(query, k=3)
            for i, res in enumerate(results, 1):
                print(f"**Kết quả {i}:**")
                print(f"Document ID: {res['document_id']}")
                print(f"Segment ID: {res['segment_id']}")
                print(f"Content: {res['content'][:300]}...\n")
            
            print("\nKết quả truy xuất multihop:")
            results = retriever.multihop_retrieve(query, k=3, m=2)
            for i, res in enumerate(results[:3], 1):
                print(f"**Kết quả {i}:**")
                print(f"Document ID: {res['document_id']}")
                print(f"Segment ID: {res['segment_id']}")
                print(f"Content: {res['content'][:300]}...\n")
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()