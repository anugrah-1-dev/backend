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

# Anthropometric ratio: infant head depth (front-to-back) / width (left-right).
# Based on neonatal/infant cephalic index studies (~78–82%). Using 0.80 as midpoint.
HEAD_DEPTH_RATIO = 0.80


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
    Measure head circumference from a FRONT-FACING photo (baby lying down,
    camera pointing at the baby's face from the front).

    Requires ArUco marker (DICT_4X4_50, 10cm×10cm) placed beside the baby's
    head on the same surface level.

    Algorithm:
      1. Detect ArUco marker → pixels_per_cm calibration.
      2. Skin segmentation (YCrCb) to isolate the head/face region.
      3. Find the largest circular skin blob near the top of the frame.
      4. Measure its horizontal bounding-box width → biparietal diameter (2a).
      5. Estimate head depth: b = a × HEAD_DEPTH_RATIO (0.80).
      6. Circumference via Ramanujan ellipse approximation.
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
                "message": "ArUco marker tidak terdeteksi. Pastikan marker 10×10 cm terlihat jelas di samping kepala bayi.",
            }

        marker_corners = corners[0][0]
        width_px = math.dist(marker_corners[0], marker_corners[1])
        height_px = math.dist(marker_corners[1], marker_corners[2])
        marker_size_px = (width_px + height_px) / 2.0

        if marker_size_px < 10:
            return {
                "success": False,
                "message": "Marker terlalu kecil. Dekatkan kamera.",
            }

        pixels_per_cm = marker_size_px / MARKER_PHYSICAL_SIZE_CM

        # ── Step 2: Mask out the marker region ─────────────────────────
        h_img, w_img = image.shape[:2]
        marker_exclude = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(marker_exclude, [marker_corners.astype(np.int32)], 255)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        marker_exclude = cv2.dilate(marker_exclude, kernel_dilate)

        # ── Step 3: Skin segmentation in YCrCb space ───────────────────
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_mask[marker_exclude > 0] = 0

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, k_close)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, k_open)

        # ── Step 4: Find the best head contour ─────────────────────────
        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "success": False,
                "message": "Kepala tidak terdeteksi. Pastikan foto dari depan dengan cahaya cukup dan wajah terlihat penuh.",
            }

        def contour_circularity(cnt):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                return 0.0
            return 4 * math.pi * area / (perimeter * perimeter)

        # Size bounds: head radius ~3–12 cm
        min_radius_cm = 3.0
        max_radius_cm = 12.0
        min_area_px = math.pi * (pixels_per_cm * min_radius_cm) ** 2
        max_area_px = math.pi * (pixels_per_cm * max_radius_cm) ** 2

        candidates = [
            cnt
            for cnt in contours
            if len(cnt) >= 5
            and min_area_px <= cv2.contourArea(cnt) <= max_area_px
            and contour_circularity(cnt) >= 0.3
        ]

        if candidates:
            # From front view, the head is typically in the upper portion of the
            # frame. Score by circularity + position (prefer higher in image).
            def head_score(cnt):
                _, cy_c, _, _ = cv2.boundingRect(cnt)
                # Normalise y to [0,1]: lower y = higher in image = better
                y_norm = cy_c / h_img
                return contour_circularity(cnt) - 0.5 * y_norm

            largest = max(candidates, key=head_score)
        else:
            size_filtered = [
                cnt
                for cnt in contours
                if len(cnt) >= 5
                and cv2.contourArea(cnt) >= min_area_px
                and cv2.contourArea(cnt) <= max_area_px
            ]
            if size_filtered:
                largest = max(size_filtered, key=cv2.contourArea)
            else:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) < 5:
                    return {
                        "success": False,
                        "message": "Kontur kepala tidak terdeteksi dengan jelas.",
                    }

        # ── Step 5: Measure biparietal width → estimate circumference ───
        # Use convex hull for a clean bounding box unaffected by concavities.
        hull = cv2.convexHull(largest)
        x_bb, _y_bb, w_bb, _h_bb = cv2.boundingRect(hull)

        # Biparietal diameter = horizontal width of head from front view.
        biparietal_px = float(w_bb)
        a_cm = (biparietal_px / 2.0) / pixels_per_cm   # semi-major axis (measured)
        b_cm = a_cm * HEAD_DEPTH_RATIO                  # semi-minor axis (estimated)

        # Ramanujan approximation: C ≈ π[3(a+b) − √((3a+b)(a+3b))]
        h_val = ((a_cm - b_cm) ** 2) / ((a_cm + b_cm) ** 2)
        circumference_cm = (
            math.pi
            * (a_cm + b_cm)
            * (1 + (3 * h_val) / (10 + math.sqrt(4 - 3 * h_val)))
        )

        warning = None
        if circumference_cm < 20 or circumference_cm > 65:
            warning = (
                f"Nilai terukur {circumference_cm:.1f} cm di luar rentang normal bayi "
                "(20–65 cm). Periksa foto dan ulangi jika perlu, atau gunakan hasil ini jika yakin."
            )

        return {
            "success": True,
            "head_circumference_cm": round(circumference_cm, 1),
            "pixels_per_cm": round(pixels_per_cm, 4),
            "biparietal_cm": round(a_cm * 2, 2),
            "estimated_depth_cm": round(b_cm * 2, 2),
            "warning": warning,
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
