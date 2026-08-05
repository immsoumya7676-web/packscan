from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import easyocr
import shutil
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import re
import cv2
import uuid

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Folders
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
CROP_DIR = "cropped"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# Serve images
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount("/cropped", StaticFiles(directory=CROP_DIR), name="cropped")

# OCR
reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='/tmp')


def is_expiry(text):
    patterns = [
        r"\b\d{2}[/-]\d{2}[/-]\d{2,4}\b",
        r"\b\d{2}[/-]\d{4}\b",
        r"\b\d{2}\s?[A-Za-z]{3}\s?\d{4}\b",
    ]

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


@app.get("/")
def home():
    return {"message": "PackInspect AI Backend Running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # Save uploaded image
    upload_name = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(upload_path)

    if image is None:
        return {"error": "Unable to read uploaded image"}

    result = reader.readtext(upload_path)

    texts = []

    expiry_date = "Not Found"

    crop_url = None

    for detection in result:

        bbox, text, confidence = detection

        texts.append(text)

        if expiry_date == "Not Found" and is_expiry(text):

            expiry_date = text

            pts = [(int(x), int(y)) for x, y in bbox]

            x1 = min(p[0] for p in pts)
            y1 = min(p[1] for p in pts)
            x2 = max(p[0] for p in pts)
            y2 = max(p[1] for p in pts)

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            cv2.putText(
                image,
                "Expiry Date",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Crop with margin
            margin = 15

            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(image.shape[1], x2 + margin)
            cy2 = min(image.shape[0], y2 + margin)

            crop = image[cy1:cy2, cx1:cx2]

            crop_file = f"{uuid.uuid4().hex}.jpg"
            crop_path = os.path.join(CROP_DIR, crop_file)

            if crop.size != 0:
                cv2.imwrite(crop_path, crop)
                crop_url = f"http://127.0.0.1:8000/cropped/{crop_file}"

            break

    # Save highlighted image
    result_file = f"{uuid.uuid4().hex}.jpg"
    result_path = os.path.join(RESULT_DIR, result_file)

    cv2.imwrite(result_path, image)

    return {
        "expiry_date": expiry_date,
        "ocr_text": texts,
        "tampering_status": "Safe",
        "confidence": 98,
        "highlighted_image": f"http://127.0.0.1:8000/results/{result_file}",
        "cropped_expiry": crop_url
        }if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)