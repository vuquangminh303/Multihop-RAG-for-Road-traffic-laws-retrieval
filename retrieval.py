import json
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# kkhởi tạo client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="",
)

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

def generate_response(query, context):
    """Sử dụng OpenRouter để tổng hợp câu trả lời từ context"""
    prompt = (
        f"Dựa trên thông tin sau đây:\n\n{context}\n\n"
        f"Hãy trả lời câu hỏi: '{query}' một cách ngắn gọn, rõ ràng và dễ hiểu."
    )
    completion = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content

def format_segment(segment):
    return (
        f"**Kết quả từ tài liệu {segment['document_id']} (Segment {segment['segment_id']})**:\n"
        f"{segment['content'][:]}...\n" 
    )

if __name__ == "__main__":
    try:
        retriever = VectorRetriever()
        query = input("Nhập câu hỏi của bạn: ")
        # lấy các đoạn văn bản liên quan
        results = retriever.retrieve(query, k=5)
        filtered_results = retriever.filter_results(query, results, threshold=0.7)
        
        if filtered_results:
            # ttổng hợp nội dung từ các kết quả tìm được
            context = "\n".join([res["content"] for res in filtered_results])
            print("\n=== KẾT QUẢ TÌM KIẾM ===")
            for i, res in enumerate(filtered_results, 1):
                print(f"{format_segment(res)}")
            
            # ttạo câu trả lời bằng llmllm
            response = generate_response(query, context)
            print("\n=== CÂU TRẢ LỜI TỔNG HỢP ===")
            print(response)
        else:
            print("Không tìm thấy kết quả nào phù hợp.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
