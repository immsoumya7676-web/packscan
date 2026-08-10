from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import easyocr
import shutil
import os
import re
import cv2
import uuid
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

print("Loading tampering detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Tampering detection model loaded successfully.")

IMG_SIZE = (224, 224)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CROP_DIR = os.path.join(BASE_DIR, "cropped")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# IMAGE PREPROCESSING
# =========================================================

def preprocess_for_ocr(image_data):

    gray = cv2.cvtColor(
        image_data,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # Upscale smaller images
    longest_side = max(h, w)

    if longest_side < 1600:

        scale = 1600 / longest_side

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Reduce noise
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# =========================================================
# IMAGE QUALITY
# =========================================================

def check_image_quality(image_data):

    gray = cv2.cvtColor(
        image_data,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    if blur_score < 25:
        blur_status = "Too blurry"

    elif blur_score < 80:
        blur_status = "Slightly blurry"

    else:
        blur_status = "Clear"

    if brightness < 45:
        lighting_status = "Too dark"

    elif brightness > 220:
        lighting_status = "Too bright"

    else:
        lighting_status = "Good lighting"

    quality_score = 100.0

    if blur_score < 25:
        quality_score -= 50

    elif blur_score < 80:
        quality_score -= 20

    if brightness < 45 or brightness > 220:
        quality_score -= 25

    quality_score = max(
        0.0,
        min(100.0, quality_score)
    )

    return {
        "score": round(
            quality_score,
            2
        ),

        "blur_score": round(
            blur_score,
            2
        ),

        "brightness": round(
            brightness,
            2
        ),

        "blur_status": blur_status,

        "lighting_status": lighting_status
    }


# =========================================================
# EXPIRY DETECTION
# =========================================================

def normalize_ocr_text(text):

    text = text.upper().strip()

    replacements = {
        "EXPIRYDATE": "EXPIRY DATE",
        "EXPDATE": "EXP DATE",
        "BESTBEFORE": "BEST BEFORE",
        "USEBY": "USE BY"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return re.sub(
        r"\s+",
        " ",
        text
    )


def extract_expiry_date(text):

    text_clean = normalize_ocr_text(
        text
    )

    patterns = [

        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",

        r"\b\d{1,2}[/-]\d{4}\b",

        r"\b\d{1,2}[/-]\d{2}\b",

        r"\b\d{1,2}\s?[A-Z]{3,9}\s?\d{2,4}\b",

        r"\b[A-Z]{3,9}\s?\d{2,4}\b",

        r"\b\d{4}[/-]\d{1,2}\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text_clean
        )

        if match:

            return match.group(0)

    return None


def expiry_score(text):

    clean = normalize_ocr_text(
        text
    )

    keywords = [

        "EXPIRY DATE",

        "EXP DATE",

        "EXPIRY",

        "BEST BEFORE",

        "USE BY",

        "EXP",

        "BBE"
    ]

    score = 0

    for keyword in keywords:

        if keyword in clean:

            score += 3

    if extract_expiry_date(clean):

        score += 4

    return score


def is_expiry(text):

    return expiry_score(text) >= 3


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

        if re.search(
            pattern,
            text_clean
        ):

            return True

    return False


# =========================================================
# SAVE CROP
# =========================================================

def save_crop(
    image_data,
    bbox,
    prefix
):

    points = [

        (int(x), int(y))

        for x, y in bbox
    ]

    x1 = min(
        p[0]
        for p in points
    )

    y1 = min(
        p[1]
        for p in points
    )

    x2 = max(
        p[0]
        for p in points
    )

    y2 = max(
        p[1]
        for p in points
    )

    margin = 25

    cx1 = max(
        0,
        x1 - margin
    )

    cy1 = max(
        0,
        y1 - margin
    )

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

    return (
        filename,
        (x1, y1, x2, y2)
    )


# =========================================================
# ANALYSIS
# =========================================================

def predict_tampering(image_data):
    resized = cv2.resize(image_data, IMG_SIZE)

    # Convert OpenCV BGR → RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Same normalization used in predict.py
    img_array = rgb.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    score = float(prediction[0][0])

    if score >= 0.65:
        status = "SUSPICIOUS"
        confidence = score * 100

    elif score <= 0.35:
        status = "SAFE"
        confidence = (1.0 - score) * 100

    else:
        status = "NEEDS REVIEW"
        confidence = max(score, 1.0 - score) * 100

    return status, round(confidence, 2), round(score, 4)


def analyze_image(
    ocr_result,
    quality
):

    confidences = [

        float(item[2])

        for item in ocr_result

        if len(item) == 3
    ]

    avg_ocr = (

        sum(confidences)
        / len(confidences)

        if confidences

        else 0.0
    )

    confidence = round(
        avg_ocr * 100,
        2
    )

    if not ocr_result:

        return (
            "UNABLE TO ANALYZE",
            confidence
        )

    if quality["score"] < 30:

        return (
            "LOW IMAGE QUALITY",
            confidence
        )

    return (
        "ANALYZED",
        confidence
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "PackInspect AI Backend Running",

        "analysis":
            "OpenCV + EasyOCR",

        "status":
            "Ready"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "ocr":
            "loaded"
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

        f"{uuid.uuid4().hex}_"
        f"{safe_filename}"
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

            "success": False,

            "error":
                "Unable to read uploaded image"
        }


    # -----------------------------------------------------
    # IMAGE QUALITY
    # -----------------------------------------------------

    quality = check_image_quality(
        image_data
    )


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    processed_image = preprocess_for_ocr(
        image_data
    )


    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    ocr_result = reader.readtext(
        processed_image
    )


    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    tampering_status, confidence, model_score = predict_tampering(
    image_data
)


    ocr_text = []

    ocr_confidences = []

    expiry_date = "Not Found"

    batch_number = "Not Found"

    expiry_crop_url = None

    batch_crop_url = None


    # =====================================================
    # PROCESS OCR
    # =====================================================

    for detection in ocr_result:

        bbox, text, ocr_confidence = detection

        text_clean = text.strip()

        if not text_clean:

            continue

        ocr_text.append(
            text_clean
        )

        ocr_confidences.append(

            round(
                float(ocr_confidence) * 100,
                2
            )
        )


        # -------------------------------------------------
        # EXPIRY
        # -------------------------------------------------

        detected_expiry_date = (
            extract_expiry_date(
                text_clean
            )
        )

        if (

            expiry_date == "Not Found"

            and expiry_score(
                text_clean
            ) >= 3
        ):

            expiry_date = (

                detected_expiry_date

                if detected_expiry_date

                else text_clean
            )

            crop_file, coordinates = (
                save_crop(
                    image_data,
                    bbox,
                    "expiry"
                )
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

                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),

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

            and is_batch(
                text_clean
            )
        ):

            batch_number = text_clean

            crop_file, coordinates = (
                save_crop(
                    image_data,
                    bbox,
                    "batch"
                )
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

                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (255, 0, 0),

                    2
                )


    # =====================================================
    # SAVE HIGHLIGHTED IMAGE
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


    average_ocr_confidence = (

        round(
            sum(ocr_confidences)
            / len(ocr_confidences),
            2
        )

        if ocr_confidences

        else 0.0
    )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "success": True,

        "expiry_date":
            expiry_date,

        "batch_number":
            batch_number,

        "ocr_text":
            ocr_text,

        "ocr_confidences":
            ocr_confidences,

        "ocr_average_confidence":
            average_ocr_confidence,

        "image_quality":
            quality,

        "tampering_status":
            tampering_status,

        "confidence":
            confidence,

        "model_score":
             model_score,

        "highlighted_image":
            f"{base_url}/results/{result_file}",

        "cropped_expiry":
            expiry_crop_url,

        "cropped_batch":
            batch_crop_url
    }


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=port
    )