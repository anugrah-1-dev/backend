# backend/main.py
# FastAPI backend for ArUco marker detection and height measurement
# ArUco marker: DICT_4X4_50, physical size 10cm × 10cm
# Run: uvicorn main:app --host 0.0.0.0 --port 8000

import base64
import math
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Posyandu Height Measurement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MARKER_PHYSICAL_SIZE_CM = 10.0  # ArUco marker is 10cm × 10cm


class MeasureRequest(BaseModel):
    image_base64: str


class CalculateRequest(BaseModel):
    head_x: float
    head_y: float
    foot_x: float
    foot_y: float
    pixels_per_cm: float
    display_width: float = 0.0
    display_height: float = 0.0


class MeasureHeadRequest(BaseModel):
    image_base64: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Posyandu Measurement API"}


@app.post("/measure")
def measure(req: MeasureRequest):
    """
    Detect ArUco marker in image and return pixels_per_cm calibration value.
    The marker physical size is 10cm × 10cm (DICT_4X4_50).
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False, "message": "Gagal mendekode gambar"}

        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ArUco detection (OpenCV 4.x API)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return {
                "success": False,
                "message": "ArUco marker tidak terdeteksi. Pastikan marker terlihat jelas dan cukup cahaya.",
            }

        # Use first detected marker
        marker_corners = corners[0][0]  # shape (4, 2)

        # Calculate marker size in pixels (average of width and height)
        width_px = math.dist(marker_corners[0], marker_corners[1])
        height_px = math.dist(marker_corners[1], marker_corners[2])
        marker_size_px = (width_px + height_px) / 2.0

        if marker_size_px < 10:
            return {
                "success": False,
                "message": "Marker terlalu kecil. Dekatkan kamera ke marker.",
            }

        pixels_per_cm = marker_size_px / MARKER_PHYSICAL_SIZE_CM

        return {
            "success": True,
            "pixels_per_cm": round(pixels_per_cm, 4),
            "marker_size_px": round(marker_size_px, 2),
            "image_width": image.shape[1],
            "image_height": image.shape[0],
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/calculate")
def calculate(req: CalculateRequest):
    """
    Calculate height from two tapped coordinates and pixels_per_cm.
    Coordinates are in display (widget) space. The backend just uses
    Euclidean distance and divides by pixels_per_cm.
    """
    try:
        dx = req.foot_x - req.head_x
        dy = req.foot_y - req.head_y
        distance_px = math.sqrt(dx * dx + dy * dy)

        if req.pixels_per_cm <= 0:
            return {"success": False, "message": "pixels_per_cm tidak valid"}

        height_cm = distance_px / req.pixels_per_cm

        if height_cm < 5 or height_cm > 200:
            return {
                "success": False,
                "message": f"Hasil tidak masuk akal ({height_cm:.1f} cm). Coba ulangi penandaan titik.",
            }

        return {
            "success": True,
            "height_cm": round(height_cm, 1),
            "distance_px": round(distance_px, 2),
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.post("/measure_head")
def measure_head(req: MeasureHeadRequest):
    """
    Measure head circumference from a top-down photo.
    Requires ArUco marker (DICT_4X4_50, 10cm×10cm) visible in the image.
    Uses skin-color segmentation + ellipse fitting (Ramanujan approximation).
    """
    try:
        image_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False, "message": "Gagal mendekode gambar"}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Detect ArUco marker for scale ──────────────────────
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return {
                "success": False,
                "message": "ArUco marker tidak terdeteksi. Pastikan marker 10×10 cm terlihat jelas di foto.",
            }

        marker_corners = corners[0][0]
        width_px = math.dist(marker_corners[0], marker_corners[1])
        height_px = math.dist(marker_corners[1], marker_corners[2])
        marker_size_px = (width_px + height_px) / 2.0

        if marker_size_px < 10:
            return {"success": False, "message": "Marker terlalu kecil. Dekatkan kamera."}

        pixels_per_cm = marker_size_px / MARKER_PHYSICAL_SIZE_CM

        # ── Step 2: Mask out the marker region ─────────────────────────
        h_img, w_img = image.shape[:2]
        marker_exclude = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(marker_exclude, [marker_corners.astype(np.int32)], 255)
        # Dilate mask slightly so marker border doesn't pollute skin detection
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        marker_exclude = cv2.dilate(marker_exclude, kernel_dilate)

        # ── Step 3: Skin segmentation in YCrCb space ───────────────────
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Broad skin range that covers light to dark skin tones
        skin_mask = cv2.inRange(ycrcb, (0, 130, 75), (255, 180, 135))

        # Remove marker area from skin mask
        skin_mask[marker_exclude > 0] = 0

        # Morphological cleanup: close small holes, open noise
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k_close)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k_open)

        # ── Step 4: Find the largest contour (head) ─────────────────────
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                "success": False,
                "message": "Kepala tidak terdeteksi. Pastikan foto diambil dari atas dengan cahaya cukup dan kepala terlihat penuh.",
            }

        largest = max(contours, key=cv2.contourArea)

        if len(largest) < 5:
            return {"success": False, "message": "Kontur kepala terlalu kecil atau tidak jelas."}

        # Area sanity check: head must be reasonably large
        min_area_px = (pixels_per_cm * 8) ** 2  # min ~8cm radius
        if cv2.contourArea(largest) < min_area_px:
            return {
                "success": False,
                "message": "Kepala terdeteksi terlalu kecil. Dekatkan kamera ke kepala bayi.",
            }

        # ── Step 5: Fit ellipse and compute circumference ───────────────
        ellipse = cv2.fitEllipse(largest)
        a_px = ellipse[1][0] / 2.0  # semi-axis 1
        b_px = ellipse[1][1] / 2.0  # semi-axis 2

        a_cm = a_px / pixels_per_cm
        b_cm = b_px / pixels_per_cm

        # Ramanujan approximation: C ≈ π[3(a+b) − √((3a+b)(a+3b))]
        h_val = ((a_cm - b_cm) ** 2) / ((a_cm + b_cm) ** 2)
        circumference_cm = math.pi * (a_cm + b_cm) * (
            1 + (3 * h_val) / (10 + math.sqrt(4 - 3 * h_val))
        )

        if circumference_cm < 25 or circumference_cm > 60:
            return {
                "success": False,
                "message": f"Hasil tidak masuk akal ({circumference_cm:.1f} cm). Pastikan foto dari tepat atas dan kepala terlihat penuh.",
            }

        return {
            "success": True,
            "head_circumference_cm": round(circumference_cm, 1),
            "pixels_per_cm": round(pixels_per_cm, 4),
            "ellipse_a_cm": round(a_cm, 2),
            "ellipse_b_cm": round(b_cm, 2),
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
