import json
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, util

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

class VectorRetriever:
    def __init__(self):
        print("Đang khởi tạo VectorRetriever...")
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        with open("metadata.json", "r", encoding="utf-8") as f:
            self.segment_data = json.load(f)
        with open("segment_ids.json", "r", encoding="utf-8") as f:
            self.segment_ids = json.load(f)
        self.index = faiss.read_index("faiss_index.bin")
        print("Đã tải xong dữ liệu!")

    def encode(self, query):
        encoded = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            return mean_pooling(self.model(**encoded), encoded["attention_mask"]).numpy()

    def retrieve(self, query, k=10):
        print(f"Đang tìm kiếm với truy vấn: {query}")
        query_embedding = self.encode(query)
        D, I = self.index.search(query_embedding, k)
        print(f"Đã tìm thấy {len(I[0])} kết quả thô")
        return [self.segment_data[i] for i in I[0] if i < len(self.segment_data)]

    def filter_results(self, query, segments, threshold=0.7):
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        query_embedding = model.encode(query)
        filtered_segments = []
        for seg in segments:
            seg_embedding = model.encode(seg["content"])
            similarity = util.pytorch_cos_sim(query_embedding, seg_embedding).item()
            if similarity >= threshold:
                filtered_segments.append(seg)
        return filtered_segments

if __name__ == "__main__":
    try:
        retriever = VectorRetriever()
        query = input("Nhập câu hỏi của bạn: ")
        results = retriever.retrieve(query, k=5)
        filtered_results = retriever.filter_results(query, results, threshold=0.7)
        if filtered_results:
            for i, res in enumerate(filtered_results, 1):
                print(f"Kết quả {i}:")
                print(f"Document ID: {res['document_id']}")
                print(f"Segment ID: {res['segment_id']}")
                print(f"Content: {res['content'][:300]}...\n")
        else:
            print("Không tìm thấy kết quả nào phù hợp.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")