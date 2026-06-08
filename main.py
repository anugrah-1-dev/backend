# backend/main.py
# FastAPI backend for ArUco marker detection and height measurement
# ArUco marker: DICT_4X4_50, physical size 10cm × 10cm
# Run: uvicorn main:app --host 0.0.0.0 --port 8000

import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Optional

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

# ── YOLO Model Cache ─────────────────────────────────────────────────────────
# Models are loaded once and reused across requests to avoid repeated disk I/O.
_yolo_models: dict = {}


def _load_yolo_model(model_name: str):
    """
    Load a YOLO pose model with in-memory caching.

    Search order:
      1. Workspace root (parent of backend/) – covers local dev with yolo11n-pose.pt
      2. Current working directory
      3. Let Ultralytics auto-download by name (e.g. "yolov8l-pose.pt")
    """
    if model_name in _yolo_models:
        return _yolo_models[model_name]

    # Import here so the server still starts even if ultralytics is not installed
    from ultralytics import YOLO  # noqa: PLC0415

    # Try workspace root first (one level above backend/)
    workspace_root = Path(__file__).parent.parent
    candidates = [workspace_root / model_name, Path(model_name)]

    model_path = next((p for p in candidates if p.exists()), model_name)
    model = YOLO(str(model_path))
    _yolo_models[model_name] = model
    return model


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


# ── Models for YOLO-based height measurement ─────────────────────────────────

class Point(BaseModel):
    """A 2-D pixel coordinate (display-space)."""
    x: float
    y: float


class MeasureHeightYoloRequest(BaseModel):
    """
    Request body for /measure_height_yolo.

    If card_p1 and card_p2 are omitted, the backend will attempt to
    auto-detect the card using OpenCV contour analysis.
    """
    image_base64: str
    # Two points the user tapped on the reference card (optional — auto-detect if absent)
    card_p1: Optional[Point] = None
    card_p2: Optional[Point] = None
    # Real-world distance between the two card points (cm)
    card_dimension_cm: float = 8.56
    # YOLO model filename.  Searched in workspace root then current dir.
    model_name: str = "yolo11n-pose.pt"
    # Minimum keypoint confidence to accept (0–1)
    min_confidence: float = 0.55


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
        # Front-facing: marker lies flat → vertical dimension is foreshortened.
        # Use the LARGEST edge (least foreshortened) for a more accurate scale.
        marker_size_px = max(width_px, height_px)

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

        # Biparietal diameter for babies: 6–14 cm → half-width in pixels
        min_bip_px = pixels_per_cm * 6.0
        max_bip_px = pixels_per_cm * 14.0

        def valid_front_head(cnt):
            """Extra filter for front-facing view: check bounding-box width
            (biparietal) and aspect ratio (head is roughly as wide as tall)."""
            if cv2.contourArea(cnt) < min_area_px or cv2.contourArea(cnt) > max_area_px:
                return False
            _x, _y, w, h = cv2.boundingRect(cnt)
            if not (min_bip_px <= w <= max_bip_px):
                return False
            aspect = w / h if h > 0 else 0
            # From front, head aspect ratio (w/h) should be 0.5 – 1.4
            return 0.5 <= aspect <= 1.4

        candidates = [
            cnt
            for cnt in contours
            if len(cnt) >= 5
            and valid_front_head(cnt)
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


# ─────────────────────────────────────────────────────────────────────────────
# /measure_height_yolo  –  YOLO Pose + reference-card scale
# ─────────────────────────────────────────────────────────────────────────────

def _detect_card_auto(image: np.ndarray) -> "Optional[float]":
    """
    Automatically detect an ATM/KTP card (ISO 7810 ID-1: 85.6 × 54 mm).
    Returns the pixel length of the long (85.6 mm) edge, or None if not found.

    Strategy: Canny edge detection → find 4-sided contours → filter by
    the card aspect ratio (1.586 ± 30 %) → pick the best candidate.
    """
    CARD_ASPECT = 85.6 / 54.0   # ≈ 1.586
    ASPECT_TOL  = 0.30           # ±30 %

    h_img, w_img = image.shape[:2]
    img_area     = h_img * w_img

    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges   = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_px: Optional[float] = None
    best_score: float        = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Card must cover 0.4 %–35 % of the image area
        if area < img_area * 0.004 or area > img_area * 0.35:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) not in (4, 5):
            continue

        _, _, bw, bh = cv2.boundingRect(approx)
        if bw == 0 or bh == 0:
            continue

        aspect = bw / bh
        if abs(aspect - CARD_ASPECT) / CARD_ASPECT <= ASPECT_TOL:
            long_px = float(bw)          # landscape: width = 85.6 mm
        elif abs(aspect - 1.0 / CARD_ASPECT) / (1.0 / CARD_ASPECT) <= ASPECT_TOL:
            long_px = float(bh)          # portrait:  height = 85.6 mm
        else:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / hull_area if hull_area > 0 else 0.0
        score     = solidity * area / img_area
        if score > best_score:
            best_score = score
            best_px    = long_px

    return best_px


# COCO 17-keypoint indices used in this endpoint
_KP_NOSE        = 0
_KP_LEFT_EYE    = 1
_KP_RIGHT_EYE   = 2
_KP_LEFT_ANKLE  = 15
_KP_RIGHT_ANKLE = 16


@app.post("/measure_height_yolo")
def measure_height_yolo(req: MeasureHeightYoloRequest):
    """
    Estimate standing height using YOLOv8/YOLO11 pose estimation.

    Workflow
    --------
    1. Decode the base64 image.
    2. Compute pixel-to-cm scale from the two card reference points
       (real-world distance = card_dimension_cm, default 8.56 cm).
    3. Load the requested YOLO pose model (cached after first call).
    4. Run inference; for each detected person:
         a. Estimate top of head from nose + eye keypoints.
         b. Select the lower ankle as the bottom reference.
         c. Compute vertical pixel span → convert to cm using scale.
    5. Return all measurements plus keypoints for client-side overlay.

    Supported models (selectable via model_name):
      - "yolo11n-pose.pt"  (default, already in workspace root)
      - "yolov8l-pose.pt"  (auto-downloaded by Ultralytics if absent)
      - any other Ultralytics pose model filename
    """
    try:
        # ── 1. Decode image ───────────────────────────────────────────
        image_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"success": False, "message": "Gagal mendekode gambar"}

        img_h, img_w = image.shape[:2]

        # ── 2. Pixel scale from reference card ───────────────────────
        if req.card_p1 is not None and req.card_p2 is not None:
            # Manual: dua titik yang dikirim oleh klien
            card_pixel_dist = math.sqrt(
                (req.card_p2.x - req.card_p1.x) ** 2
                + (req.card_p2.y - req.card_p1.y) ** 2
            )
            if card_pixel_dist < 10:
                return {
                    "success": False,
                    "message": (
                        "Jarak titik referensi kartu terlalu kecil "
                        "(< 10 px). Ketuk dua sudut kartu yang berjauhan."
                    ),
                }
        else:
            # Auto: deteksi kartu dari kontur gambar
            card_pixel_dist = _detect_card_auto(image)
            if card_pixel_dist is None:
                return {
                    "success": False,
                    "message": (
                        "Kartu ATM/KTP tidak terdeteksi otomatis. "
                        "Pastikan kartu terlihat jelas, datar, "
                        "dan pencahayaan cukup."
                    ),
                }

        cm_per_pixel = req.card_dimension_cm / card_pixel_dist
        pixels_per_cm = card_pixel_dist / req.card_dimension_cm

        # ── 3. Load YOLO model ────────────────────────────────────────
        try:
            model = _load_yolo_model(req.model_name)
        except Exception as model_err:
            return {
                "success": False,
                "message": f"Gagal memuat model YOLO '{req.model_name}': {model_err}",
            }

        # ── 4. Run YOLO inference ─────────────────────────────────────
        results = model(image, verbose=False)

        if not results or results[0].keypoints is None:
            return {"success": False, "message": "Tidak ada orang terdeteksi di gambar"}

        kps_data = results[0].keypoints.data  # (N, 17, 3): [x, y, conf]
        boxes    = results[0].boxes

        if len(kps_data) == 0:
            return {"success": False, "message": "Tidak ada orang terdeteksi di gambar"}

        # ── 5. Compute height for each detected person ────────────────
        measurements = []
        min_conf = req.min_confidence

        for i, kps_tensor in enumerate(kps_data):
            kps = kps_tensor.cpu().numpy()  # (17, 3)

            nose_x,  nose_y,  nose_c  = kps[_KP_NOSE]
            leye_x,  leye_y,  leye_c  = kps[_KP_LEFT_EYE]
            reye_x,  reye_y,  reye_c  = kps[_KP_RIGHT_EYE]
            lank_x,  lank_y,  lank_c  = kps[_KP_LEFT_ANKLE]
            rank_x,  rank_y,  rank_c  = kps[_KP_RIGHT_ANKLE]

            # Nose must be visible
            if float(nose_c) < min_conf:
                continue

            # ── Estimate top of head ──────────────────────────────────
            # Eyes sit roughly at mid-face height.  The crown of the head
            # is approximately the same distance above the eyes as the
            # eyes are above the nose.
            eye_ys = []
            if float(leye_c) >= min_conf:
                eye_ys.append(float(leye_y))
            if float(reye_c) >= min_conf:
                eye_ys.append(float(reye_y))

            if eye_ys:
                avg_eye_y    = sum(eye_ys) / len(eye_ys)
                eye_nose_gap = float(nose_y) - avg_eye_y  # > 0 when nose is below eyes
                # Crown ≈ eyes_y − eye_nose_gap (clipped to image top)
                head_top_y = max(0.0, avg_eye_y - max(eye_nose_gap, 0.0))
            else:
                # Fallback: no eye keypoints → use nose directly
                head_top_y = float(nose_y)

            # ── Select bottom reference (lower ankle) ─────────────────
            ankle_ys = []
            if float(lank_c) >= min_conf:
                ankle_ys.append(float(lank_y))
            if float(rank_c) >= min_conf:
                ankle_ys.append(float(rank_y))

            if not ankle_ys:
                # Person detected but ankles not visible → skip
                continue

            # Higher y-value = lower in image = closer to the ground
            ankle_y = max(ankle_ys)

            # ── Height calculation ────────────────────────────────────
            height_px = ankle_y - head_top_y

            if height_px < 50:
                # Implausibly short span in pixels → likely a detection artefact
                continue

            height_cm = height_px * cm_per_pixel

            # Sanity-check: accept heights between 30 cm (infant) and 250 cm
            if not (30.0 <= height_cm <= 250.0):
                continue

            # ── Bounding box ──────────────────────────────────────────
            bbox = None
            if boxes is not None and i < len(boxes):
                b = boxes[i].xyxy[0].cpu().numpy()
                bbox = {
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                }

            # ── Visible keypoints for client overlay ──────────────────
            visible_kps = [
                {
                    "index": k_idx,
                    "x": float(kps[k_idx][0]),
                    "y": float(kps[k_idx][1]),
                    "confidence": round(float(kps[k_idx][2]), 3),
                }
                for k_idx in range(17)
                if float(kps[k_idx][2]) >= min_conf
            ]

            measurements.append(
                {
                    "person_index": i,
                    "height_cm": round(float(height_cm), 1),
                    "height_px": round(float(height_px), 1),
                    "head_top_y": round(float(head_top_y), 1),
                    "ankle_y": round(float(ankle_y), 1),
                    "nose_confidence": round(float(nose_c), 3),
                    "bbox": bbox,
                    "keypoints": visible_kps,
                }
            )

        if not measurements:
            return {
                "success": False,
                "message": (
                    "Tidak dapat menghitung tinggi badan. "
                    "Pastikan seluruh tubuh (kepala & kaki) terlihat di frame "
                    f"dengan confidence ≥ {int(min_conf * 100)}%."
                ),
            }

        return {
            "success": True,
            "pixels_per_cm": round(float(pixels_per_cm), 4),
            "cm_per_pixel": round(float(cm_per_pixel), 6),
            "card_pixel_dist": round(float(card_pixel_dist), 2),
            "card_dimension_cm": req.card_dimension_cm,
            "model_used": req.model_name,
            "image_width": img_w,
            "image_height": img_h,
            "person_count": len(measurements),
            "measurements": measurements,
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}
