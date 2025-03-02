#!/usr/bin/env python3
"""
Script để cập nhật huggingface_hub và transformers đến phiên bản mới nhất.
Giải quyết lỗi: ImportError: cannot import name 'hf_hub_url' from 'huggingface_hub.utils'
"""

import os
import sys
import subprocess
import pkg_resources

def check_python_version():
    """Kiểm tra phiên bản Python."""
    print(f"Phiên bản Python: {sys.version}")
    if sys.version_info < (3, 8):
        print("CẢNH BÁO: Python 3.8 trở lên được khuyến nghị.")
    return True

def check_pip_installation():
    """Kiểm tra xem pip đã được cài đặt chưa."""
    try:
        subprocess.check_output([sys.executable, '-m', 'pip', '--version'])
        return True
    except subprocess.CalledProcessError:
        print("Lỗi: pip chưa được cài đặt.")
        return False

def get_package_version(package_name):
    """Lấy phiên bản của gói đã cài đặt, trả về None nếu chưa cài đặt."""
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return None

def run_command(cmd):
    """Chạy lệnh và hiển thị output theo thời gian thực."""
    print(f"Đang chạy: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Lỗi khi chạy lệnh: {cmd}")
        return False
    return True

def get_latest_version(package_name):
    """Lấy phiên bản mới nhất của gói từ PyPI."""
    cmd = f"{sys.executable} -m pip index versions {package_name}"
    process = subprocess.Popen(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, _ = process.communicate()
    
    if process.returncode != 0:
        print(f"Không thể lấy thông tin phiên bản cho {package_name}")
        return None
    
    # Tìm dòng có chứa "Available versions"
    for line in stdout.split('\n'):
        if "Available versions" in line:
            # Lấy phiên bản đầu tiên (mới nhất)
            versions = line.split(':')[1].strip().split(',')
            if versions:
                return versions[0].strip()
    
    return None

def update_package(package_name, version=None):
    """Cập nhật gói đến phiên bản cụ thể hoặc mới nhất."""
    current_version = get_package_version(package_name)
    
    if version:
        version_spec = f"{package_name}=={version}"
        print(f"Cập nhật {package_name} từ {current_version or 'không có'} đến {version}")
    else:
        version_spec = package_name
        latest = get_latest_version(package_name)
        print(f"Cập nhật {package_name} từ {current_version or 'không có'} đến phiên bản mới nhất{' (' + latest + ')' if latest else ''}")
    
    return run_command(f"{sys.executable} -m pip install --upgrade {version_spec}")

def uninstall_package(package_name):
    """Gỡ bỏ gói."""
    if get_package_version(package_name):
        print(f"Gỡ bỏ {package_name}")
        return run_command(f"{sys.executable} -m pip uninstall -y {package_name}")
    print(f"{package_name} chưa được cài đặt.")
    return True

def check_import(module_name, from_module=None, import_name=None):
    """Kiểm tra xem có thể import module/function cụ thể không."""
    if from_module and import_name:
        try:
            exec(f"from {from_module} import {import_name}")
            print(f"✓ Import thành công: from {from_module} import {import_name}")
            return True
        except ImportError as e:
            print(f"✗ Lỗi import: from {from_module} import {import_name}")
            print(f"  {str(e)}")
            return False
    else:
        try:
            exec(f"import {module_name}")
            print(f"✓ Import thành công: import {module_name}")
            return True
        except ImportError as e:
            print(f"✗ Lỗi import: import {module_name}")
            print(f"  {str(e)}")
            return False

def fix_huggingface_hub_error():
    """Khắc phục lỗi không thể import hf_hub_url từ huggingface_hub.utils."""
    print("\n=== Khắc phục lỗi huggingface_hub ===")
    
    # Kiểm tra phiên bản hiện tại
    hf_hub_version = get_package_version("huggingface_hub")
    transformers_version = get_package_version("transformers")
    
    print(f"Phiên bản huggingface_hub hiện tại: {hf_hub_version or 'chưa cài đặt'}")
    print(f"Phiên bản transformers hiện tại: {transformers_version or 'chưa cài đặt'}")
    
    # Kiểm tra lỗi import
    can_import_hf_hub_url = check_import("huggingface_hub.utils", "huggingface_hub.utils", "hf_hub_url")
    
    if can_import_hf_hub_url:
        print("✅ Không phát hiện lỗi import. huggingface_hub.utils.hf_hub_url hoạt động bình thường.")
        return True
    
    print("\nĐang khắc phục lỗi...")
    
    # Gỡ bỏ sentence-transformers nếu có
    uninstall_package("sentence-transformers")
    
    # Cập nhật huggingface_hub trước
    print("\n1. Cập nhật huggingface_hub lên phiên bản mới nhất")
    update_package("huggingface_hub")
    
    # Cập nhật transformers
    print("\n2. Cập nhật transformers lên phiên bản mới nhất")
    update_package("transformers")
    
    # Cài đặt lại các thư viện cần thiết cho hệ thống
    print("\n3. Cài đặt torch")
    update_package("torch")
    
    # Kiểm tra lại
    print("\n=== Kiểm tra sau khi khắc phục ===")
    hf_hub_version = get_package_version("huggingface_hub")
    transformers_version = get_package_version("transformers")
    
    print(f"Phiên bản huggingface_hub mới: {hf_hub_version}")
    print(f"Phiên bản transformers mới: {transformers_version}")
    
    can_import_hf_hub_url = check_import("huggingface_hub.utils", "huggingface_hub.utils", "hf_hub_url")
    
    if not can_import_hf_hub_url:
        print("\n⚠️ Vẫn không thể import hf_hub_url. Thử biện pháp khác...")
        print("Cài đặt huggingface_hub phiên bản cụ thể (0.16.4)")
        update_package("huggingface_hub", "0.16.4")
        
        can_import_hf_hub_url = check_import("huggingface_hub.utils", "huggingface_hub.utils", "hf_hub_url")
        
        if not can_import_hf_hub_url:
            print("\n❌ Không thể khắc phục lỗi import hf_hub_url.")
            print("Vui lòng thử tạo môi trường ảo mới và cài đặt lại các thư viện.")
            return False
    
    print("\n✅ Đã khắc phục lỗi import hf_hub_url thành công!")
    return True

def main():
    """Hàm chính."""
    print("=== Công cụ cập nhật và khắc phục lỗi huggingface_hub ===")
    
    check_python_version()
    
    if not check_pip_installation():
        print("Vui lòng cài đặt pip trước khi tiếp tục.")
        sys.exit(1)
    
    fix_huggingface_hub_error()
    
    print("\n=== Hoàn tất cập nhật ===")
    print("Bạn có thể chạy lại các script như sau:")
    print("1. python retrieval.py - để kiểm tra hệ thống truy vấn")
    print("2. python server.py - để khởi động API server")
    print("3. python multihop_rag.py - để kiểm tra hệ thống RAG")

if __name__ == "__main__":
    main() 