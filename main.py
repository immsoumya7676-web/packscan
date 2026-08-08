from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import easyocr
import shutil
import os
import re
import cv2
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# =========================================================
# SETTINGS
# =========================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CROP_DIR = os.path.join(BASE_DIR, "cropped")

MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="PackInspect AI",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# STATIC FILES
# =========================================================

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


# =========================================================
# LOAD TRAINED TAMPERING MODEL
# =========================================================

print("Loading tampering detection model...")

tampering_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Tampering model loaded successfully.")


# =========================================================
# LOAD OCR
# =========================================================

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False,
    model_storage_directory="/tmp"
)

print("EasyOCR loaded successfully.")


# =========================================================
# EXPIRY DETECTION
# =========================================================

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

    for keyword in expiry_keywords:
        if keyword in text_clean:
            return True

    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{1,2}[/-]\d{4}\b",
        r"\b\d{1,2}[/-]\d{2}\b",
        r"\b\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4}\b",
        r"\b[A-Za-z]{3,9}\s?\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}\b"
    ]

    for pattern in date_patterns:
        if re.search(pattern, text_clean):
            return True

    return False


# =========================================================
# BATCH DETECTION
# =========================================================

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
        "MFG BATCH"
    ]

    for keyword in batch_keywords:
        if keyword in text_clean:
            return True

    batch_patterns = [
        r"\bB[0-9A-Z]{3,}\b",
        r"\bLOT[ -]?[0-9A-Z]{2,}\b",
        r"\b[A-Z]{1,4}[ -]?[0-9]{3,8}\b"
    ]

    for pattern in batch_patterns:
        if re.search(pattern, text_clean):
            return True

    return False


# =========================================================
# SAVE OCR CROP
# =========================================================

def save_crop(image_data, bbox, prefix):

    points = [
        (int(x), int(y))
        for x, y in bbox
    ]

    x1 = min(p[0] for p in points)
    y1 = min(p[1] for p in points)

    x2 = max(p[0] for p in points)
    y2 = max(p[1] for p in points)

    margin = 25

    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)

    cx2 = min(
        image_data.shape[1],
        x2 + margin
    )

    cy2 = min(
        image_data.shape[0],
        y2 + margin
    )

    crop = image_data[
        cy1:cy2,
        cx1:cx2
    ]

    if crop.size == 0:
        return None, None

    filename = (
        f"{prefix}_{uuid.uuid4().hex}.jpg"
    )

    filepath = os.path.join(
        CROP_DIR,
        filename
    )

    cv2.imwrite(
        filepath,
        crop
    )

    return filename, (x1, y1, x2, y2)


# =========================================================
# TAMPERING PREDICTION
# =========================================================

def predict_tampering(image_path):

    IMG_SIZE = (224, 224)

    # Same preprocessing used in predict.py
    img = image.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = tampering_model.predict(
        img_array,
        verbose=0
    )

    # Your predict.py uses class 0
    # as the tampered probability.
    tampered_probability = float(
        prediction[0][0]
    )

    if tampered_probability > 0.5:

        result = "TAMPERED"

        confidence = (
            tampered_probability * 100
        )

    else:

        result = "ORIGINAL"

        confidence = (
            (1 - tampered_probability) * 100
        )

    return result, round(confidence, 2)


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "PackInspect AI Backend Running",
        "model": "Loaded",
        "status": "Ready"
    }


# =========================================================
# ANALYZE
# =========================================================

@app.post("/analyze")
async def analyze(
    request: Request,
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # SAVE UPLOAD
    # -----------------------------------------------------

    safe_filename = (
        file.filename
        or "uploaded_image.jpg"
    )

    upload_name = (
        f"{uuid.uuid4().hex}_{safe_filename}"
    )

    upload_path = os.path.join(
        UPLOAD_DIR,
        upload_name
    )

    with open(
        upload_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    image_data = cv2.imread(
        upload_path
    )

    if image_data is None:

        return {
            "error": "Unable to read uploaded image"
        }


    # =====================================================
    # 1. ACTUAL TRAINED MODEL PREDICTION
    # =====================================================

    tampering_status, confidence = (
        predict_tampering(upload_path)
    )


    # =====================================================
    # 2. OCR
    # =====================================================

    ocr_result = reader.readtext(
        upload_path
    )

    ocr_text = []

    expiry_date = "Not Found"
    batch_number = "Not Found"

    expiry_crop_url = None
    batch_crop_url = None


    # =====================================================
    # 3. PROCESS OCR
    # =====================================================

    for detection in ocr_result:

        bbox, text, ocr_confidence = detection

        text_clean = text.strip()

        if not text_clean:
            continue

        ocr_text.append(
            text_clean
        )


        # -------------------------------------------------
        # EXPIRY
        # -------------------------------------------------

        if (
            expiry_date == "Not Found"
            and is_expiry(text_clean)
        ):

            expiry_date = text_clean

            crop_file, coordinates = save_crop(
                image_data,
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
                    image_data,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    image_data,
                    "EXPIRY",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )


        # -------------------------------------------------
        # BATCH
        # -------------------------------------------------

        if (
            batch_number == "Not Found"
            and is_batch(text_clean)
        ):

            batch_number = text_clean

            crop_file, coordinates = save_crop(
                image_data,
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
                    image_data,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    image_data,
                    "BATCH",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )


    # =====================================================
    # 4. SAVE HIGHLIGHTED IMAGE
    # =====================================================

    result_file = (
        f"{uuid.uuid4().hex}.jpg"
    )

    result_path = os.path.join(
        RESULT_DIR,
        result_file
    )

    cv2.imwrite(
        result_path,
        image_data
    )

    base_url = (
        str(request.base_url)
        .rstrip("/")
    )


    # =====================================================
    # 5. RETURN REAL RESULT
    # =====================================================

    return {

        "expiry_date": expiry_date,

        "batch_number": batch_number,

        "ocr_text": ocr_text,

        "tampering_status": tampering_status,

        "confidence": confidence,

        "highlighted_image": (
            f"{base_url}/results/{result_file}"
        ),

        "cropped_expiry": expiry_crop_url,

        "cropped_batch": batch_crop_url
    }