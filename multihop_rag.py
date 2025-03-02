from retrieval import VectorRetriever

class MultiHopRAG:
    def __init__(self):
        try:
            self.retriever = VectorRetriever()
            print("Khởi tạo VectorRetriever thành công!")
        except Exception as e:
            print(f"Lỗi khi khởi tạo VectorRetriever: {e}")
            raise

    def answer_query(self, query, k=5):
        segments = self.retriever.retrieve(query, k)
        if not segments:
            return {"answer": "Không tìm thấy thông tin phù hợp.", "references": []}
        answer = "Dựa trên thông tin:\n"
        for i, seg in enumerate(segments, 1):
            # Chỉ lấy 100 ký tự đầu tiên làm ví dụ tóm tắt
            summary = seg['content'][:100] + "..." if len(seg['content']) > 100 else seg['content']
            answer += f"{i}. {summary}\n"
        return {"answer": answer, "references": segments}

if __name__ == "__main__":
    rag = MultiHopRAG()
    query = input("Nhập câu hỏi của bạn: ")
    result = rag.answer_query(query)
    print(result["answer"])