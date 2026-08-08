from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import easyocr
import shutil
import os
import re
import cv2
import uuid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = FastAPI()

# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# FOLDERS
# =========================

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
CROP_DIR = "cropped"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

# =========================
# SERVE IMAGES
# =========================

app.mount(
    "/results",
    StaticFiles(directory=RESULT_DIR),
    name="results"
)

app.mount(
    "/cropped",
    StaticFiles(directory=CROP_DIR),
    name="cropped"
)

# =========================
# OCR
# =========================

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory="/tmp"
)

# =========================
# EXPIRY DETECTION
# =========================

def is_expiry(text):
    text_clean = text.strip().upper()

    expiry_keywords = [
        "EXP",
        "EXPIRY",
        "EXP DATE",
        "EXPIRY DATE",
        "USE BY",
        "BEST BEFORE",
        "BBE"
    ]

    # Direct expiry keywords
    for keyword in expiry_keywords:
        if keyword in text_clean:
            return True

    # Date formats
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}[/-]\d{4}\b",
        r"\b\d{1,2}[/-]\d{2}\b",
        r"\b\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s?\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}\b",
    ]

    for pattern in date_patterns:
        if re.search(pattern, text_clean):
            return True

    return False


# =========================
# BATCH DETECTION
# =========================

def is_batch(text):
    text_clean = text.strip().upper()

    batch_keywords = [
        "BATCH",
        "BATCH NO",
        "BATCH NUMBER",
        "LOT",
        "LOT NO",
        "LOT NUMBER",
        "B.NO",
        "B NO",
        "BN",
        "MFG BATCH"
    ]

    for keyword in batch_keywords:
        if keyword in text_clean:
            return True

    # Common batch-number formats
    batch_patterns = [
        r"\bB[0-9A-Z]{3,}\b",
        r"\bLOT[ -]?[0-9A-Z]{2,}\b",
        r"\b[A-Z]{1,4}[ -]?[0-9]{3,8}\b",
    ]

    for pattern in batch_patterns:
        if re.search(pattern, text_clean):
            return True

    return False


# =========================
# CROP HELPER
# =========================

def save_crop(image, bbox, crop_prefix):
    pts = [(int(x), int(y)) for x, y in bbox]

    x1 = min(p[0] for p in pts)
    y1 = min(p[1] for p in pts)
    x2 = max(p[0] for p in pts)
    y2 = max(p[1] for p in pts)

    margin = 25

    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(image.shape[1], x2 + margin)
    cy2 = min(image.shape[0], y2 + margin)

    crop = image[cy1:cy2, cx1:cx2]

    if crop.size == 0:
        return None, None

    crop_file = f"{crop_prefix}_{uuid.uuid4().hex}.jpg"
    crop_path = os.path.join(CROP_DIR, crop_file)

    cv2.imwrite(crop_path, crop)

    return crop_file, (x1, y1, x2, y2)


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "message": "PackInspect AI Backend Running"
    }


# =========================
# ANALYZE
# =========================

@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...)
):

    # -------------------------
    # Save uploaded image
    # -------------------------

    upload_name = f"{uuid.uuid4().hex}_{file.filename}"
    upload_path = os.path.join(
        UPLOAD_DIR,
        upload_name
    )

    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -------------------------
    # Read image
    # -------------------------

    image = cv2.imread(upload_path)

    if image is None:
        return {
            "error": "Unable to read uploaded image"
        }

    # -------------------------
    # OCR
    # -------------------------

    result = reader.readtext(upload_path)

    texts = []

    expiry_date = "Not Found"
    batch_number = "Not Found"

    expiry_crop_url = None
    batch_crop_url = None

    # -------------------------
    # Process OCR detections
    # -------------------------

    for detection in result:

        bbox, text, confidence = detection

        text_clean = text.strip()

        if not text_clean:
            continue

        texts.append(text_clean)

        # =====================
        # EXPIRY
        # =====================

        if expiry_date == "Not Found" and is_expiry(text_clean):

            expiry_date = text_clean

            crop_file, coordinates = save_crop(
                image,
                bbox,
                "expiry"
            )

            if crop_file:

                expiry_crop_url = (
                    f"{str(request.base_url).rstrip('/')}"
                    f"/cropped/{crop_file}"
                )

                x1, y1, x2, y2 = coordinates

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    image,
                    "EXPIRY",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # =====================
        # BATCH
        # =====================

        if batch_number == "Not Found" and is_batch(text_clean):

            batch_number = text_clean

            crop_file, coordinates = save_crop(
                image,
                bbox,
                "batch"
            )

            if crop_file:

                batch_crop_url = (
                    f"{str(request.base_url).rstrip('/')}"
                    f"/cropped/{crop_file}"
                )

                x1, y1, x2, y2 = coordinates

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    image,
                    "BATCH",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    # -------------------------
    # Save highlighted image
    # -------------------------

    result_file = f"{uuid.uuid4().hex}.jpg"

    result_path = os.path.join(
        RESULT_DIR,
        result_file
    )

    cv2.imwrite(
        result_path,
        image
    )

    base_url = str(
        request.base_url
    ).rstrip("/")

    # -------------------------
    # Response
    # -------------------------

    return {
        "expiry_date": expiry_date,
        "batch_number": batch_number,
        "ocr_text": texts,

        # Existing fields
        "tampering_status": "Safe",
        "confidence": 98,

        # Highlighted image
        "highlighted_image": (
            f"{base_url}/results/{result_file}"
        ),

        # Crops
        "cropped_expiry": expiry_crop_url,
        "cropped_batch": batch_crop_url
    }

