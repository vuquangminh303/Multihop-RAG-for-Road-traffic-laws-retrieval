from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import faiss
import json
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# Khởi tạo OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Định nghĩa request body
class QueryRequest(BaseModel):
    query: str

# VectorRetriever class (moved here for completeness)
class VectorRetriever:
    def __init__(self):
        print("Starting VectorRetriever...")
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        with open("metadata.json", "r", encoding="utf-8") as f:
            self.segment_data = json.load(f)
        with open("segment_ids.json", "r", encoding="utf-8") as f:
            self.segment_ids = json.load(f)
        self.index = faiss.read_index("faiss_index.bin")
        print("Data loaded!")

    def encode(self, query):
        encoded = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded)
            mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(output[0].size()).float()
            return torch.sum(output[0] * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9).numpy()

    def retrieve(self, query, k=5):
        query_embedding = self.encode(query)
        D, I = self.index.search(query_embedding, k)
        return [self.segment_data[i] for i in I[0] if i < len(self.segment_data)]

    def filter_results(self, query, segments, threshold=0.7):
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        query_embedding = model.encode(query)
        filtered = []
        for seg in segments:
            seg_embedding = model.encode(seg["content"])
            similarity = util.pytorch_cos_sim(query_embedding, seg_embedding).item()
            if similarity >= threshold:
                filtered.append(seg)
        return filtered

# Helper functions
def generate_response(query, context):
    prompt = f"Dựa trên thông tin sau đây:\n\n{context}\n\nHãy trả lời câu hỏi: '{query}' ngắn gọn, rõ ràng."
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",  # Swap if needed
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Couldn’t generate response: {str(e)}"

def format_segment(segment):
    return f"**From doc {segment['document_id']} (Segment {segment['segment_id']})**:\n{segment['content'][:200]}..."

# Initialize retriever once
retriever = VectorRetriever()

# Routes
@app.get("/")
async def root():
    return {"message": "Server’s live! POST to /search with a query."}

@app.post("/search")
async def search(query_request: QueryRequest):
    query = query_request.query
    results = retriever.retrieve(query, k=5)
    filtered_results = retriever.filter_results(query, results, threshold=0.7)
    
    if filtered_results:
        context = "\n".join([res["content"] for res in filtered_results])
        response = generate_response(query, context)
        formatted_results = [format_segment(res) for res in filtered_results]
        return {
            "results": formatted_results,
            "answer": response
        }
    return {"message": "No matching results found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)