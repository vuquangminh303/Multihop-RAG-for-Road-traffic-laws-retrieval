import os
import json
from docx import Document
from pdfminer.high_level import extract_text as pdf_extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Hàm trích xuất văn bản từ file
def extract_text(file_path):
    print(f"Processing {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".pdf":
        return pdf_extract_text(file_path)
    return ""

# Hàm chia văn bản thành các đoạn
def split_into_segments(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    return splitter.split_text(text)

# Hàm xử lý các file trong thư mục
def process_files(directory):
    all_segments = []
    files = [f for f in os.listdir(directory) if f.endswith((".docx", ".pdf"))]
    for file in files:
        file_path = os.path.join(directory, file)
        text = extract_text(file_path)
        segments = split_into_segments(text)
        doc_id = os.path.splitext(file)[0]
        for i, segment in enumerate(segments):
            all_segments.append({
                "document_id": doc_id,
                "segment_id": f"{doc_id}_{i}",
                "content": segment
            })
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False)

# Chạy script
directory = "/home/cong/workspace/chatbot-retrieval-based/Thongtu"
process_files(directory)