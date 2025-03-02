from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional, List, Dict, Any
from multihop_rag import MultiHopRAG

# Khởi tạo FastAPI app
app = FastAPI(
    title="MultiHop RAG API",
    description="API cho hệ thống Retrieval-Augmented Generation nhiều bước, hỗ trợ tra cứu thông tin từ văn bản pháp luật.",
    version="1.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo RAG hệ thống
# Mặc định không load LLM để giảm yêu cầu phần cứng, có thể bật bằng cách set use_llm=True
rag = MultiHopRAG(use_llm=False)

# Định nghĩa model cho request và response
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    m: Optional[int] = 3
    use_multihop: Optional[bool] = True
    max_context_tokens: Optional[int] = 3000

class Reference(BaseModel):
    document_id: str
    title: str
    issuing_agency: str
    date: str
    segments: List[Dict[str, str]]

class QueryResponse(BaseModel):
    answer: str
    references: List[Reference]

@app.get("/")
async def root():
    return {"message": "MultiHop RAG API đang hoạt động. Truy cập /docs để xem tài liệu API."}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Truy vấn và trả lời dựa trên văn bản pháp luật
    
    - **query**: Câu hỏi hoặc truy vấn của người dùng
    - **k**: Số lượng đoạn văn bản truy xuất trong lần đầu tiên (mặc định: 5)
    - **m**: Số lượng đoạn văn bản truy xuất cho mỗi đoạn văn bản ở bước 1 (mặc định: 3)
    - **use_multihop**: Sử dụng truy xuất nhiều bước hoặc truy xuất đơn giản (mặc định: True)
    - **max_context_tokens**: Số token tối đa cho văn bản ngữ cảnh (mặc định: 3000)
    """
    try:
        result = rag.answer_query(
            request.query,
            k=request.k,
            m=request.m,
            use_multihop=request.use_multihop,
            max_context_tokens=request.max_context_tokens
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý truy vấn: {str(e)}")

@app.get("/health")
async def health_check():
    """Kiểm tra trạng thái hoạt động của API"""
    return {"status": "healthy"}

if __name__ == "__main__":
    print("Khởi động MultiHop RAG API server...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 