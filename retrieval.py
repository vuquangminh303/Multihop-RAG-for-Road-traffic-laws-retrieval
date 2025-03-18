import json
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import numpy as np
# kkhởi tạo client OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
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
        self.model = SentenceTransformer(model_name)
        with open("metadata.json", "r", encoding="utf-8") as f:
            self.segment_data = json.load(f)
        with open("segment_ids.json", "r", encoding="utf-8") as f:
            self.segment_ids = json.load(f)
        self.index = faiss.read_index("faiss_index.bin")
        print("Đã tải xong dữ liệu!")
    def process_query(self,query):
        prompt = f"""Viết lại query sau đây của người dùng thành một yêu cầu rõ ràng, cụ thể và trang trọng bằng tiếng Việt, phù hợp để truy xuất thông tin từ một cơ sở dữ liệu vector.
    Nếu như query có liên quan đến nhiều thông tin hãy tách ra làm nhiều query mới để có thể truy xuất nhiều lần. Với mỗi query hãy cho bắt đầu và kết thúc nằm ở trong '*'.
    Chỉ trả về query mới, không cần giải thích gì thêm
    Lưu ý rằng query mới sẽ được gửi đến cơ sở dữ liệu vector, nơi thực hiện tìm kiếm tương đồng để truy xuất tài liệu. Ví dụ:  
    Query: Chúng tôi có một bài luận phải nộp vào ngày mai. Chúng tôi phải viết về một số loài động vật. Tôi yêu chim cánh cụt. Tôi có thể viết về chúng. Nhưng tôi cũng có thể viết về cá heo. Chúng có phải là động vật không? Có lẽ vậy. Hãy viết về cá heo. Ví dụ, chúng sống ở đâu?  
    Answer: * Cá heo sống ở đâu *
    Ví dụ:  
    Query: So sánh doanh thu của FPT và Viettel
    Answer : * Doanh thu của FPT *,* Doanh thu Viettel*
    Bây giờ, hãy viết lại query sau: Query: {query} Answer:"""
        completion = client.chat.completions.create(
            model="google/gemma-3-27b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        results = completion.choices[0].message.content
        queries = [line.strip('* ').strip() for line in results.splitlines() if line.strip()]
        return queries
    def encode(self, queries):
        queries_embedding = []
        for query in queries:
            query_embedding = self.model.encode(query)
            query_embedding = query_embedding.astype(np.float32)  # Chuyển sang float32
            queries_embedding.append(query_embedding)
        return queries_embedding

    def retrieve(self, queries, k=10):
        print(f"Đang tìm kiếm với truy vấn: {queries}")
        queries_embedding = self.encode(queries)
        results = {}
        for idx, query in enumerate(queries):
            # Tạo mảng numpy 2D từ embedding
            embedding = np.expand_dims(queries_embedding[idx], axis=0)
            D, I = self.index.search(embedding, k)
            print(f"Truy vấn '{query}' tìm thấy {len(I[0])} kết quả thô")
            valid_indices = [i for i in I[0] if i < len(self.segment_data)]
            results[query] = [self.segment_data[i] for i in valid_indices]
        return results

    def filter_results(self, query, segments, threshold=0):
        query_embedding = self.model.encode(query)
        filtered_segments = []
        for seg in segments:
            seg_embedding = self.model.encode(seg["content"])
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
        processed_queries = retriever.process_query(query)
        results = retriever.retrieve(processed_queries, k=5)
        
        # Thu thập tất cả các segments từ kết quả
        all_segments = []
        for seg_list in results.values():
            all_segments.extend(seg_list)
        
        # Lọc kết quả cho từng query
        filtered_results = []
        for q in processed_queries:
            filtered = retriever.filter_results(q, all_segments, threshold=0.55)
            filtered_results.extend(filtered)
        
        if filtered_results:
            context = "\n".join([res["content"] for res in filtered_results])
            print("\n=== KẾT QUẢ TÌM KIẾM ===")
            for i, res in enumerate(filtered_results, 1):
                print(f"{format_segment(res)}")
            
            response = generate_response(query, context)
            print("\n=== CÂU TRẢ LỜI TỔNG HỢP ===")
            print(response)
        else:
            print("Không tìm thấy kết quả nào phù hợp.")
    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")