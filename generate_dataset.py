import os
import cv2
import random
import numpy as np
import albumentations as A

# -----------------------------
# FOLDERS
# -----------------------------
INPUT_FOLDER = "datasets/original"
OUTPUT_FOLDER = "datasets/altered"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Number of augmented images per original
NUM_AUG = 20

# -----------------------------
# ALBUMENTATIONS PIPELINE
# -----------------------------
transform = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.25,
        contrast_limit=0.25,
        p=0.8
),

    A.GaussianBlur(
        blur_limit=(3,7),
        p=0.4
    ),

    A.MotionBlur(
        blur_limit=7,
        p=0.3
    ),

    A.GaussNoise(
    std_range=(0.02, 0.08),
    p=0.4
),


    A.Rotate(
        limit=5,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5
    ),

    A.Affine(
        scale=(0.95,1.05),
        translate_percent=(-0.02,0.02),
        rotate=(-2,2),
        p=0.5
    ),

    A.Perspective(
        scale=(0.02,0.05),
        p=0.3
    ),

    A.ImageCompression(
        quality_range=(45,90),
        p=0.5
    )
])

# -----------------------------
# CUSTOM EFFECTS
# -----------------------------

def add_shadow(img):
    overlay = img.copy()
    h, w = img.shape[:2]

    x = random.randint(0, w // 2)

    cv2.rectangle(
        overlay,
        (x, 0),
        (w, h),
        (0, 0, 0),
        -1
    )

    return cv2.addWeighted(overlay, 0.25, img, 0.75, 0)


def add_glare(img):
    overlay = img.copy()

    h, w = img.shape[:2]

    center = (
        random.randint(0, w),
        random.randint(0, h)
    )

    radius = random.randint(40, 120)

    cv2.circle(
        overlay,
        center,
        radius,
        (255,255,255),
        -1
    )

    return cv2.addWeighted(
        overlay,
        0.15,
        img,
        0.85,
        0
    )


def add_scratches(img):
    out = img.copy()

    h, w = out.shape[:2]

    for _ in range(random.randint(2,6)):
        x1 = random.randint(0,w)
        y1 = random.randint(0,h)

        x2 = random.randint(0,w)
        y2 = random.randint(0,h)

        color = random.randint(180,255)

        cv2.line(
            out,
            (x1,y1),
            (x2,y2),
            (color,color,color),
            1
        )

    return out


def add_dust(img):
    out = img.copy()

    h, w = out.shape[:2]

    for _ in range(random.randint(80,180)):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)

        r = random.randint(1,2)

        cv2.circle(
            out,
            (x,y),
            r,
            (random.randint(170,255),)*3,
            -1
        )

    return out


def fade(img):
    white = np.full_like(img,255)

    return cv2.addWeighted(
        img,
        0.88,
        white,
        0.12,
        0
    )


# -----------------------------
# GENERATION
# -----------------------------

effects = [
    add_shadow,
    add_glare,
    add_scratches,
    add_dust,
    fade
]

count = 0

for file in os.listdir(INPUT_FOLDER):

    path = os.path.join(INPUT_FOLDER,file)

    img = cv2.imread(path)

    if img is None:
        continue

    name = os.path.splitext(file)[0]

    for i in range(NUM_AUG):

        aug = transform(image=img)["image"]

        random.shuffle(effects)

        for effect in effects[:random.randint(1,3)]:
            aug = effect(aug)

        save_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_aug_{i+1}.jpg"
        )

        cv2.imwrite(save_path, aug)

        count += 1

print("="*40)
print("Dataset Generation Complete")
print(f"Images Generated : {count}")
print("="*40)