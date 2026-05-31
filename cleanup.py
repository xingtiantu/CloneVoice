#!/usr/bin/env python3
"""清理本地缓存：uploads 和 models 目录"""
import os
import shutil

def clean_dir(path, keep_patterns=None):
    if not os.path.exists(path):
        return
    keep_patterns = keep_patterns or []
    for item in os.listdir(path):
        if any(p in item for p in keep_patterns):
            continue
        item_path = os.path.join(path, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"  删除目录: {item}")
            else:
                os.remove(item_path)
                print(f"  删除文件: {item}")
        except Exception as e:
            print(f"  跳过 {item}: {e}")

if __name__ == "__main__":
    print("[Clean] 清理 uploads/ ...")
    clean_dir("uploads")
    print("[Clean] 清理 models/ ...")
    clean_dir("models")
    print("[Clean] 清理完成，刷新网页即可重新开始")
