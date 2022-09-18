from fastapi import FastAPI, UploadFile
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from starlette.responses import FileResponse
from src.express import Xpress
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

app = FastAPI()


@app.post("/upload-file/")
def save_upload_file_tmp(upload_file: UploadFile) -> FileResponse:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()

    global path
    path = f"{tmp_path}"

    return FileResponse(path)


@app.get("/images/")
def emotions():
    test_image_one = plt.imread(path)
    emo_detector = Xpress(mtcnn=True)

    captured_emotions = emo_detector.detect_emotions(test_image_one)
    emotions_dict = [d['emotions'] for d in captured_emotions][0]

    return emotions_dict

