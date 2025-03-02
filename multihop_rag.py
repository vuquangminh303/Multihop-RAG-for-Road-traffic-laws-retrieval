import json
import re
import numpy as np
from retrieval import VectorRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

class MultiHopRAG:
    def __init__(self, retriever=None, use_llm=True, llm_model_name="vinai/PhoGPT-7B5-Instruct"):
        # Khởi tạo retriever nếu không được cung cấp
        print("Đang khởi tạo MultiHop RAG...")
        try:
            self.retriever = retriever if retriever else VectorRetriever()
            
            # Khởi tạo LLM và tokenizer nếu sử dụng
            self.use_llm = use_llm
            if use_llm:
                try:
                    print(f"Đang tải mô hình LLM: {llm_model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_model_name, 
                        device_map="auto", 
                        load_in_8bit=True if "PhoGPT" in llm_model_name else False
                    )
                    print("Đã tải LLM thành công!")
                except Exception as e:
                    print(f"Lỗi khi tải LLM: {e}")
                    print("Tiếp tục mà không có LLM...")
                    self.use_llm = False
        except Exception as e:
            print(f"Lỗi khi khởi tạo MultiHop RAG: {e}")
            raise
                
    def retrieve_documents(self, query, k=5, m=3, use_multihop=True):
        """Retrieve relevant segments using either simple or multihop retrieval"""
        try:
            if use_multihop:
                segments = self.retriever.multihop_retrieve(query, k=k, m=m)
            else:
                segments = self.retriever.simple_retrieve(query, k=k)
            return segments
        except Exception as e:
            print(f"Lỗi khi truy xuất tài liệu: {e}")
            return []
        
    def group_by_document(self, segments):
        """Group segments by document and sort by segment order"""
        docs = defaultdict(list)
        for segment in segments:
            doc_id = segment['document_id']
            docs[doc_id].append(segment)
            
        # Sort segments within each document
        for doc_id in docs:
            docs[doc_id].sort(key=self.get_segment_number)
            
        return docs
    
    def get_segment_number(self, segment):
        """Extract segment number from segment_id"""
        segment_id = segment['segment_id']
        match = re.search(r'_(\d+)$', segment_id)
        if match:
            return int(match.group(1))
        return 0
        
    def format_references(self, segments):
        """Format segments into reference documents"""
        docs = self.group_by_document(segments)
        references = []
        
        for doc_id, segs in docs.items():
            # Get metadata from first segment
            doc_info = {
                'document_id': doc_id,
                'title': segs[0]['title'],
                'issuing_agency': segs[0]['issuing_agency'],
                'date': segs[0]['date'],
                'segments': []
            }
            
            # Add segments
            for seg in segs:
                doc_info['segments'].append({
                    'segment_id': seg['segment_id'],
                    'content': seg['content']
                })
                
            references.append(doc_info)
            
        return references
    
    def generate_prompt(self, query, segments, max_tokens=3000):
        """Generate a prompt for the LLM based on retrieved segments"""
        prompt = f"""Bạn là trợ lý AI chuyên về pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên thông tin sau:

Câu hỏi: {query}

Thông tin tham khảo:
"""
        # Add context from segments
        total_tokens = 0
        for i, segment in enumerate(segments):
            segment_text = f"[{segment['document_id']}]: {segment['content']}"
            segment_tokens = len(segment_text.split())
            
            if total_tokens + segment_tokens > max_tokens:
                # Truncate if adding this segment would exceed max_tokens
                remaining_tokens = max_tokens - total_tokens
                words = segment_text.split()
                truncated_text = ' '.join(words[:remaining_tokens])
                prompt += truncated_text + "...\n\n"
                break
            
            prompt += segment_text + "\n\n"
            total_tokens += segment_tokens
            
        prompt += """
Dựa vào thông tin được cung cấp, hãy trả lời câu hỏi một cách ngắn gọn, chính xác và đầy đủ. 
Nếu thông tin không đủ để trả lời, hãy thừa nhận rằng bạn không biết. 
Không được tạo ra thông tin không có trong tài liệu tham khảo.

Trả lời:
"""
        return prompt
    
    def answer_query(self, query, k=5, m=3, use_multihop=True, max_context_tokens=3000):
        """Answer a query using RAG"""
        try:
            # Retrieve relevant segments
            segments = self.retrieve_documents(query, k, m, use_multihop)
            
            if not segments:
                return {
                    "answer": "Không tìm thấy thông tin liên quan. Vui lòng thử lại với câu hỏi khác.",
                    "references": []
                }
            
            # Format references
            references = self.format_references(segments)
            
            # Generate answer using LLM if available
            if self.use_llm:
                try:
                    prompt = self.generate_prompt(query, segments, max_context_tokens)
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                    
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    
                    answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                except Exception as e:
                    print(f"Lỗi khi tạo câu trả lời với LLM: {e}")
                    answer = self._create_simple_answer(segments)
            else:
                answer = self._create_simple_answer(segments)
            
            return {
                "answer": answer,
                "references": references
            }
        except Exception as e:
            print(f"Lỗi khi trả lời truy vấn: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Đã xảy ra lỗi khi xử lý truy vấn: {str(e)}",
                "references": []
            }
    
    def _create_simple_answer(self, segments, max_segments=3):
        """Create a simple answer without LLM by concatenating top segments"""
        answer = "Dựa trên các tài liệu tham khảo, tôi tìm thấy những thông tin sau:\n\n"
        
        for i, segment in enumerate(segments[:max_segments]):
            answer += f"- {segment['content'][:500]}...\n\n"
            
        return answer
            
if __name__ == "__main__":
    # Test
    rag = MultiHopRAG(use_llm=False)
    result = rag.answer_query("Quy định về tốc độ tối đa của xe máy là gì?")
    print(result["answer"]) 