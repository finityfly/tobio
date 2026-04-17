import os
import shutil

from fastapi import UploadFile


def save_temp_video(file: UploadFile, cache_dir: str = "cache") -> str:
    temp_path = os.path.join(cache_dir, f"temp_{file.filename}")
    os.makedirs(cache_dir, exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path
