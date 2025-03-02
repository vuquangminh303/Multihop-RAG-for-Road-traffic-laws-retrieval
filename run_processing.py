#!/usr/bin/env python3
import os
import subprocess
import sys
import time

def print_header(message):
    print("\n" + "=" * 80)
    print(f" {message} ".center(80, "="))
    print("=" * 80)

def run_command(command, exit_on_error=True):
    print(f"\n>>> Executing: {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # In output theo thời gian thực
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')
    
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0 and exit_on_error:
        print(f"Command failed with return code {return_code}")
        sys.exit(1)
    
    return return_code == 0

def main():
    print_header("HƯỚNG DẪN XỬ LÝ DỮ LIỆU THÔNG TƯ")
    print("\nChương trình này sẽ hướng dẫn bạn qua các bước xử lý dữ liệu:")
    print("1. Cài đặt các thư viện cần thiết")
    print("2. Tạo metadata từ các file PDF/DOCX")
    print("3. Tạo vector embeddings")
    print("4. Kiểm tra hệ thống retrieval")
    
    input("\nNhấn Enter để bắt đầu...")
    
    # Bước 1: Cài đặt thư viện
    print_header("BƯỚC 1: CÀI ĐẶT THƯ VIỆN")
    print("\nCài đặt các thư viện cần thiết...")
    choice = input("Bạn có muốn cài đặt các thư viện cần thiết không? (y/n, mặc định: y): ") or "y"
    if choice.lower() == "y":
        run_command("pip install -r requirements.txt")
    
    # Bước 2: Tạo metadata
    print_header("BƯỚC 2: TẠO METADATA")
    print("\nTrích xuất metadata từ các file PDF/DOCX trong thư mục Thongtu...")
    choice = input("Bạn có muốn chạy bước này không? (y/n, mặc định: y): ") or "y"
    if choice.lower() == "y":
        if not os.path.exists("create_metadata.py"):
            print("Lỗi: File create_metadata.py không tồn tại!")
        else:
            run_command("python create_metadata.py")
    
    # Bước 3: Tạo embeddings
    print_header("BƯỚC 3: TẠO VECTOR EMBEDDINGS")
    print("\nTạo và lưu vector embeddings cho các đoạn văn bản...")
    if not os.path.exists("metadata.json"):
        print("Lỗi: File metadata.json không tồn tại. Hãy chạy bước 2 trước!")
    else:
        choice = input("Bạn có muốn chạy bước này không? (y/n, mặc định: y): ") or "y"
        if choice.lower() == "y":
            if os.path.exists("create_embeddings_fixed.py"):
                run_command("python create_embeddings_fixed.py")
            else:
                print("Lỗi: File create_embeddings_fixed.py không tồn tại!")
                print("Đã tìm thấy các phiên bản thay thế:")
                alternatives = [f for f in os.listdir(".") if f.startswith("create_embeddings") and f.endswith(".py")]
                if alternatives:
                    print("\nCác phiên bản có sẵn:")
                    for i, alt in enumerate(alternatives):
                        print(f"{i+1}. {alt}")
                    choice = input("\nChọn một phiên bản để chạy (nhập số, nhấn Enter để bỏ qua): ")
                    if choice.isdigit() and 1 <= int(choice) <= len(alternatives):
                        selected = alternatives[int(choice)-1]
                        run_command(f"python {selected}")
    
    # Bước 4: Kiểm tra retrieval
    print_header("BƯỚC 4: KIỂM TRA HỆ THỐNG RETRIEVAL")
    print("\nKiểm tra hệ thống truy xuất thông tin...")
    if not os.path.exists("faiss_index.bin") or not os.path.exists("segment_ids.json"):
        print("Lỗi: File faiss_index.bin hoặc segment_ids.json không tồn tại. Hãy chạy bước 3 trước!")
    else:
        choice = input("Bạn có muốn chạy bước này không? (y/n, mặc định: y): ") or "y"
        if choice.lower() == "y":
            if not os.path.exists("retrieval.py"):
                print("Lỗi: File retrieval.py không tồn tại!")
            else:
                run_command("python retrieval.py")
    
    # Hoàn tất
    print_header("QUÁ TRÌNH XỬ LÝ ĐÃ HOÀN TẤT")
    print("\nTất cả các bước đã được thực hiện. Bạn có thể chạy API server bằng lệnh:")
    print("python server.py")
    print("\nHoặc sử dụng trực tiếp hệ thống MultiHop RAG trong Python:")
    print("from multihop_rag import MultiHopRAG")
    print("rag = MultiHopRAG(use_llm=False)")
    print("result = rag.answer_query('Quy định về tốc độ tối đa của xe máy là gì?')")
    print("\nCảm ơn bạn đã sử dụng hệ thống này!")

if __name__ == "__main__":
    main() 