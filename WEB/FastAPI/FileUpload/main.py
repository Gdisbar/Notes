from typing import Annotated
import os
import mimetypes
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()


# @app.post("/files/")
# async def create_files(files: Annotated[list[bytes], File()]):
#     return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    upload_dir = "./uploads"  # Directory to store uploaded files
    os.makedirs(upload_dir, exist_ok=True)

    uploaded_files = {}
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        uploaded_files[f"name : {file.filename} "] = [
            f"content_type : {file.content_type},",
            f"file_size : {len(content) * 0.001} Kb",
        ]
    return uploaded_files

@app.get("/filelists")
async def select_uploaded_file():
    target_path = "./uploads"
    uploaded_files = {}
    for file in os.listdir(target_path):
        file_path = os.path.join(target_path,file)
        print(file_path)
        content_type,_ = mimetypes.guess_type(file)
        uploaded_files[f"name : {file} "] = [
            f"content_type : {content_type},",
            f"file_size : {round(len(file) * 0.001,3)} Kb",
        ]
    return uploaded_files

