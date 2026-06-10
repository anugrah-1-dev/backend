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

# Anthropometric ratio: head AP depth (front-to-back) / width (left-right).
# Cephalic index CI = W/AP ≈ 78–82% (mean ~80%). So AP/W = 1/CI ≈ 1.25.
# HEAD_DEPTH_RATIO = AP/W = 1.25 (depth is LONGER than width for most humans).
HEAD_DEPTH_RATIO = 1.25

# ── Head top estimation constants ────────────────────────────────────────────
# Jarak ubun-ubun ke mata ≈ 2.5× jarak mata ke hidung (studi antropometri).
# Pada bayi/balita kepala relatif lebih besar, sehingga faktor sedikit lebih tinggi.
HEAD_TOP_MULTIPLIER = 2.5

# Offset pergelangan kaki → telapak kaki (cm).
# Keypoint "ankle" YOLO ada di malleolus, bukan telapak kaki.
# Estimasi konservatif: 5–7 cm; gunakan 6 cm sebagai nilai tengah.
ANKLE_TO_FOOT_CM = 6.0

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


class Point(BaseModel):
    """A 2-D pixel coordinate (display-space)."""
    x: float
    y: float


class MeasureHeadRequest(BaseModel):
    image_base64: str
    # Reference card for scale (optional — auto-detect if absent)
    card_p1: Optional[Point] = None
    card_p2: Optional[Point] = None
    card_dimension_cm: float = 8.56
    # YOLO segmentation model — gunakan nano agar tidak OOM di Railway
    seg_model_name: str = "yolov8n-seg.pt"
    min_confidence: float = 0.50


# ── Models for YOLO-based height measurement ─────────────────────────────────

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
    Estimate head circumference using YOLOv8 segmentation + ellipse fitting.

    Scale calibration: ATM/KTP card long edge (default 8.56 cm) — same
    approach as /measure_height_yolo.  Optionally supply two card corner
    points (card_p1 / card_p2); omit for auto-detect via contour analysis.

    Algorithm
    ---------
    1. Compute cm_per_pixel from the reference card.
    2. Run yolov8n-seg.pt segmentation; pick the tallest person detection
       (COCO class 0 = person).
    3. Crop the segmentation mask to the top HEAD_FRACTION of the person
       bounding box (the head region).
    4. Find contours → filter by minimum area (head radius ≥ 3 cm).
    5. Fit an ellipse (cv2.fitEllipse) to the largest valid contour.
    6. Validate: head must be roughly front-facing (minor/major axis ≥ 0.55).
    7. Ramanujan circumference: C ≈ π × √(2(a² + b²)).
    8. Return result + ellipse coordinates for Flutter overlay.
    """
    # Fraction of person bbox height (from top) treated as the head region.
    # ~22 % covers both adult heads (~1/8 body) and infant heads (~1/4 body).
    HEAD_FRACTION = 0.22

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

        cm_per_pixel  = req.card_dimension_cm / card_pixel_dist
        pixels_per_cm = card_pixel_dist / req.card_dimension_cm

        # ── 3. Load YOLO segmentation model ──────────────────────────
        try:
            seg_model = _load_yolo_model(req.seg_model_name)
        except Exception as model_err:
            return {
                "success": False,
                "message": (
                    f"Gagal memuat model segmentasi '{req.seg_model_name}': {model_err}"
                ),
            }

        # ── 4. Run segmentation ───────────────────────────────────────
        results = seg_model(image, verbose=False)

        if (
            not results
            or results[0].masks is None
            or results[0].boxes is None
        ):
            return {
                "success": False,
                "message": "Tidak ada orang terdeteksi di gambar.",
            }

        masks_data = results[0].masks.data.cpu().numpy()  # (N, H_m, W_m)
        boxes      = results[0].boxes
        classes    = boxes.cls.cpu().numpy()
        confs      = boxes.conf.cpu().numpy()
        xyxy_all   = boxes.xyxy.cpu().numpy()  # (N, 4)

        # Filter: COCO class 0 = person, minimum confidence
        person_ids = [
            i
            for i, (cls, conf) in enumerate(zip(classes, confs))
            if int(cls) == 0 and float(conf) >= req.min_confidence
        ]

        if not person_ids:
            return {
                "success": False,
                "message": "Tidak ada orang terdeteksi dengan confidence yang cukup.",
            }

        # Pick the person with the tallest bounding box (likely the subject)
        best_idx = max(person_ids, key=lambda i: xyxy_all[i][3] - xyxy_all[i][1])

        x1, y1, x2, y2 = xyxy_all[best_idx]
        bbox_h = y2 - y1

        # ── 5. Build head mask from segmentation ─────────────────────
        raw_mask = masks_data[best_idx]  # float 0–1, shape (H_m, W_m)
        # Resize mask to full image resolution
        mask_resized = cv2.resize(
            raw_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )
        mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255

        # Crop to head region: top HEAD_FRACTION of the person bbox
        hx1 = max(0, int(x1))
        hy1 = max(0, int(y1))
        hx2 = min(img_w, int(x2))
        hy2 = min(img_h, int(y1 + bbox_h * HEAD_FRACTION))

        head_mask = mask_bin[hy1:hy2, hx1:hx2]

        if head_mask.size == 0 or np.sum(head_mask) == 0:
            return {
                "success": False,
                "message": (
                    "Area kepala tidak ditemukan di mask segmentasi. "
                    "Pastikan kepala terlihat penuh dalam frame."
                ),
            }

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, k_close)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_OPEN,  k_open)

        # ── 6. Find contours ──────────────────────────────────────────
        contours, _ = cv2.findContours(
            head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {
                "success": False,
                "message": (
                    "Kontur kepala tidak ditemukan. "
                    "Pastikan kepala terlihat jelas dengan pencahayaan cukup."
                ),
            }

        # Minimum contour area: head radius ≥ 3 cm
        min_area_px = math.pi * (pixels_per_cm * 3.0) ** 2

        valid_cnts = [
            c for c in contours if len(c) >= 5 and cv2.contourArea(c) >= min_area_px
        ]

        if not valid_cnts:
            return {
                "success": False,
                "message": (
                    "Kontur kepala terlalu kecil. "
                    "Dekatkan kamera atau pastikan kepala terlihat penuh."
                ),
            }

        best_cnt = max(valid_cnts, key=cv2.contourArea)

        # ── 7. Fit ellipse ────────────────────────────────────────────
        try:
            ellipse = cv2.fitEllipse(best_cnt)
        except cv2.error:
            return {
                "success": False,
                "message": (
                    "Gagal fitting ellipse ke kontur kepala. "
                    "Coba dengan foto yang lebih jelas."
                ),
            }

        (cx_crop, cy_crop), (MA_px, ma_px), angle_deg = ellipse

        # Map ellipse centre back to full-image coordinates
        cx_full = float(cx_crop) + hx1
        cy_full = float(cy_crop) + hy1

        a_px = MA_px / 2.0  # semi-major axis (pixels)
        b_px = ma_px / 2.0  # semi-minor axis (pixels)

        # ── 8. Validate: head facing forward ─────────────────────────
        if a_px <= 0 or b_px <= 0:
            return {"success": False, "message": "Dimensi ellipse tidak valid."}

        axis_ratio = min(a_px, b_px) / max(a_px, b_px)
        if axis_ratio < 0.50:
            return {
                "success": False,
                "message": (
                    "Kepala terdeteksi miring atau tidak menghadap lurus ke depan. "
                    "Pastikan wajah menghadap kamera secara langsung."
                ),
            }

        # ── 9. Circumference via Ramanujan approximation ─────────────
        #
        # Dari foto depan, axis minor ellipse = tinggi kepala yang terlihat,
        # bukan kedalaman (AP) kepala. Lingkar kepala sesungguhnya membutuhkan
        # lebar (W) dan kedalaman (AP = HEAD_DEPTH_RATIO × W).
        #
        # cv2.fitEllipse angle: sudut sumbu mayor terhadap sumbu-x (horizontal).
        #   angle ≈ 0° / 180° → sumbu mayor vertikal → sumbu minor = lebar kepala
        #   angle ≈ 90°       → sumbu mayor horizontal → sumbu mayor = lebar kepala
        angle_mod = float(angle_deg) % 180.0
        if 45.0 <= angle_mod <= 135.0:
            # angle ≈ 90° → sumbu mayor VERTIKAL (tinggi kepala)
            # → lebar kepala = sumbu MINOR = b_px
            width_semi_px = b_px
        else:
            # angle ≈ 0°/180° → sumbu mayor HORIZONTAL (lebar kepala)
            # → lebar kepala = sumbu mayor = a_px
            width_semi_px = a_px

        # Semi-axis lebar (lateral) dan estimasi kedalaman (AP)
        a_cm = width_semi_px * cm_per_pixel          # lebar/2
        b_cm = a_cm * HEAD_DEPTH_RATIO               # kedalaman/2 ≈ 0.80 × lebar/2

        # C ≈ π × √(2(a² + b²))
        circumference_cm = math.pi * math.sqrt(2.0 * (a_cm ** 2 + b_cm ** 2))

        warning = None
        if circumference_cm < 25.0 or circumference_cm > 65.0:
            warning = (
                f"Nilai terukur {circumference_cm:.1f} cm di luar rentang normal "
                "(25–65 cm). Periksa foto dan ulangi jika perlu."
            )

        return {
            "success": True,
            "head_circumference_cm": round(circumference_cm, 1),
            "pixels_per_cm": round(pixels_per_cm, 4),
            "semi_major_cm": round(a_cm, 2),
            "semi_minor_cm": round(b_cm, 2),
            "ellipse": {
                "center_x": round(cx_full, 1),
                "center_y": round(cy_full, 1),
                "semi_major_px": round(a_px, 1),
                "semi_minor_px": round(b_px, 1),
                "angle_deg": round(float(angle_deg), 1),
            },
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

    Strategy (multi-pass):
    Pass 1 – Adaptive threshold on grayscale (works for light-coloured cards).
    Pass 2 – Canny edge detection (fallback for dark/coloured cards).
    Both passes filter by card aspect ratio (1.586 ± 35%) and area (0.1%–45%).
    """
    CARD_ASPECT = 85.6 / 54.0   # ≈ 1.586
    ASPECT_TOL  = 0.12           # ±12 % — cukup ketat agar wajah tidak ikut terdeteksi

    h_img, w_img = image.shape[:2]
    img_area     = h_img * w_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _find_card_in_edges(edge_img: np.ndarray) -> "Optional[float]":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edge_img, kernel, iterations=1)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_px: Optional[float] = None
        best_score: float        = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Card must cover 0.1 %–45 % of the image area
            if area < img_area * 0.001 or area > img_area * 0.45:
                continue

            peri   = cv2.arcLength(cnt, True)
            # Longgarkan epsilon supaya kontur bertekstur tetap terbaca sebagai 4-sisi
            for eps_factor in (0.02, 0.04, 0.06):
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                if len(approx) in (4, 5):
                    break
            else:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw == 0 or bh == 0:
                continue

            aspect = bw / bh
            if abs(aspect - CARD_ASPECT) / CARD_ASPECT <= ASPECT_TOL:
                long_px = float(bw)
            elif abs(aspect - 1.0 / CARD_ASPECT) / (1.0 / CARD_ASPECT) <= ASPECT_TOL:
                long_px = float(bh)
            else:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity  = area / hull_area if hull_area > 0 else 0.0
            # Kartu ATM berbentuk persegi panjang (solidity tinggi ≥ 0.75)
            # Wajah/oval memiliki solidity lebih rendah
            if solidity < 0.75:
                continue
            score     = solidity * area / img_area
            if score > best_score:
                best_score = score
                best_px    = long_px

        return best_px

    # ── Pass 1: Adaptive threshold (kartu terang di background gelap/beragam) ──
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 51, -10
    )
    result = _find_card_in_edges(thresh)
    if result is not None:
        return result

    # ── Pass 2: Otsu threshold ────────────────────────────────────────────────
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = _find_card_in_edges(otsu)
    if result is not None:
        return result

    # ── Pass 3: Canny (fallback untuk kartu gelap/berwarna) ───────────────────
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 20, 80)
    return _find_card_in_edges(edges)


# COCO 17-keypoint indices
_KP_NOSE        = 0
_KP_LEFT_EYE    = 1
_KP_RIGHT_EYE   = 2
_KP_LEFT_EAR    = 3
_KP_RIGHT_EAR   = 4
_KP_LEFT_ANKLE  = 15
_KP_RIGHT_ANKLE = 16


def _estimate_head_top(
    nose_y: float,
    nose_c: float,
    leye_y: float,
    leye_c: float,
    reye_y: float,
    reye_c: float,
    lear_y: float,
    lear_c: float,
    rear_y: float,
    rear_c: float,
    img_h: int,
    min_conf: float,
) -> float:
    """
    Estimasi posisi Y ubun-ubun (crown of head) dari keypoint wajah.

    Strategi (urutan prioritas):
    1. Jika mata terdeteksi → gunakan jarak mata-hidung × HEAD_TOP_MULTIPLIER
       untuk memperkirakan ubun-ubun di atas mata.
    2. Jika hanya telinga terdeteksi → ubun-ubun ≈ telinga − (nose−ear gap).
    3. Fallback → hidung dikurangi estimasi kasar berbasis tinggi gambar.

    Kenapa HEAD_TOP_MULTIPLIER = 2.5?
    - Jarak mata ke hidung (eye-nose gap) pada manusia ≈ 1/3 tinggi wajah.
    - Jarak mata ke ubun-ubun ≈ 2/3 tinggi wajah (lebih besar dari gap mata-hidung).
    - Rasio (2/3) / (1/3) = 2.0 secara teori; faktor 2.5 memberi buffer karena
      keypoint "nose" YOLO sering berada sedikit di bawah pangkal hidung,
      sehingga eye-nose gap underestimated.
    - Pada bayi/balita kepala proporsional lebih tinggi → faktor 2.5 lebih aman
      daripada 2.0.
    """
    eye_ys = []
    if leye_c >= min_conf:
        eye_ys.append(leye_y)
    if reye_c >= min_conf:
        eye_ys.append(reye_y)

    if eye_ys:
        avg_eye_y    = sum(eye_ys) / len(eye_ys)
        eye_nose_gap = nose_y - avg_eye_y   # positif → hidung di bawah mata

        # Ubun-ubun = mata - (HEAD_TOP_MULTIPLIER × jarak mata-hidung)
        crown_offset = max(eye_nose_gap, 0.0) * HEAD_TOP_MULTIPLIER
        head_top_y   = avg_eye_y - crown_offset
        return max(0.0, head_top_y)

    # Fallback: gunakan telinga jika tersedia
    ear_ys = []
    if lear_c >= min_conf:
        ear_ys.append(lear_y)
    if rear_c >= min_conf:
        ear_ys.append(rear_y)

    if ear_ys:
        avg_ear_y    = sum(ear_ys) / len(ear_ys)
        ear_nose_gap = nose_y - avg_ear_y   # biasanya ≈ 0 (hidung & telinga sejajar)
        # Ubun-ubun ≈ telinga ke atas sejauh 1.5× jarak telinga-hidung
        crown_offset = max(ear_nose_gap, 0.0) * 1.5 + abs(ear_nose_gap) * 0.5
        head_top_y   = avg_ear_y - crown_offset
        return max(0.0, head_top_y)

    # Fallback terakhir: hidung dikurangi ~8% tinggi gambar
    fallback_offset = img_h * 0.08
    return max(0.0, nose_y - fallback_offset)


@app.post("/measure_height_yolo")
def measure_height_yolo(req: MeasureHeightYoloRequest):
    """
    Estimasi tinggi badan berdiri menggunakan YOLO Pose + skala kartu referensi.

    Perbaikan v2 (bug-fix dari versi sebelumnya):
    ──────────────────────────────────────────────
    1. HEAD_TOP_MULTIPLIER = 2.5
       Versi lama memakai eye_nose_gap × 1.0 untuk offset ubun-ubun,
       sehingga head_top_y terlalu rendah → tinggi badan under-estimated ~60%.
       Faktor 2.5 sesuai rasio antropometri ubun-ubun:mata:hidung.

    2. Koreksi ankle → telapak kaki (ANKLE_TO_FOOT_CM = 6 cm)
       Keypoint "ankle" YOLO ada di malleolus (pergelangan), bukan telapak kaki.
       Tanpa koreksi ini, tinggi kehilangan ~5–7 cm.

    3. Fallback bertingkat untuk head_top_y
       Jika mata tidak terdeteksi, gunakan telinga; jika tidak ada keduanya,
       gunakan hidung dengan offset berbasis resolusi gambar.

    Workflow
    ────────
    1. Decode gambar base64.
    2. Hitung skala pixel/cm dari dua titik kartu (atau auto-detect).
    3. Load model YOLO pose (cached).
    4. Jalankan inferensi; untuk setiap orang yang terdeteksi:
         a. Estimasi ubun-ubun dari keypoint hidung + mata + telinga.
         b. Ambil ankle yang lebih rendah + offset ke telapak kaki.
         c. Hitung span vertikal piksel → konversi ke cm.
    5. Kembalikan semua pengukuran + keypoint untuk overlay di klien.
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

        cm_per_pixel  = req.card_dimension_cm / card_pixel_dist
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

            nose_x,  nose_y,  nose_c  = float(kps[_KP_NOSE][0]),  float(kps[_KP_NOSE][1]),  float(kps[_KP_NOSE][2])
            leye_x,  leye_y,  leye_c  = float(kps[_KP_LEFT_EYE][0]),  float(kps[_KP_LEFT_EYE][1]),  float(kps[_KP_LEFT_EYE][2])
            reye_x,  reye_y,  reye_c  = float(kps[_KP_RIGHT_EYE][0]), float(kps[_KP_RIGHT_EYE][1]), float(kps[_KP_RIGHT_EYE][2])
            lear_x,  lear_y,  lear_c  = float(kps[_KP_LEFT_EAR][0]),  float(kps[_KP_LEFT_EAR][1]),  float(kps[_KP_LEFT_EAR][2])
            rear_x,  rear_y,  rear_c  = float(kps[_KP_RIGHT_EAR][0]), float(kps[_KP_RIGHT_EAR][1]), float(kps[_KP_RIGHT_EAR][2])
            lank_x,  lank_y,  lank_c  = float(kps[_KP_LEFT_ANKLE][0]),  float(kps[_KP_LEFT_ANKLE][1]),  float(kps[_KP_LEFT_ANKLE][2])
            rank_x,  rank_y,  rank_c  = float(kps[_KP_RIGHT_ANKLE][0]), float(kps[_KP_RIGHT_ANKLE][1]), float(kps[_KP_RIGHT_ANKLE][2])

            # Hidung wajib terdeteksi sebagai anchor wajah
            if nose_c < min_conf:
                continue

            # ── FIX 1: Estimasi ubun-ubun dengan multiplier antropometri ─
            head_top_y = _estimate_head_top(
                nose_y=nose_y, nose_c=nose_c,
                leye_y=leye_y, leye_c=leye_c,
                reye_y=reye_y, reye_c=reye_c,
                lear_y=lear_y, lear_c=lear_c,
                rear_y=rear_y, rear_c=rear_c,
                img_h=img_h,
                min_conf=min_conf,
            )

            # ── FIX 2: Titik bawah = ankle + offset ke telapak kaki ───
            ankle_ys = []
            if lank_c >= min_conf:
                ankle_ys.append(lank_y)
            if rank_c >= min_conf:
                ankle_ys.append(rank_y)

            if not ankle_ys:
                # Ankle tidak terdeteksi → lewati orang ini
                continue

            # y terbesar = posisi paling bawah di gambar
            ankle_y = max(ankle_ys)

            # Konversi offset fisik ke piksel dan tambahkan ke ankle_y
            ankle_to_foot_px = ANKLE_TO_FOOT_CM * pixels_per_cm
            foot_bottom_y    = min(float(img_h - 1), ankle_y + ankle_to_foot_px)

            # ── Height calculation ────────────────────────────────────
            height_px = foot_bottom_y - head_top_y

            if height_px < 50:
                # Span terlalu kecil → artefak deteksi
                continue

            height_cm = height_px * cm_per_pixel

            # Sanity-check: terima tinggi 30–250 cm
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
                    "foot_bottom_y": round(float(foot_bottom_y), 1),
                    "ankle_to_foot_px": round(float(ankle_to_foot_px), 1),
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


# ─────────────────────────────────────────────────────────────────────────────
# /debug_card  –  Debug deteksi kartu (kirim gambar, lihat semua kandidat)
# ─────────────────────────────────────────────────────────────────────────────

class DebugCardRequest(BaseModel):
    image_base64: str


@app.post("/debug_card")
def debug_card(req: DebugCardRequest):
    """
    Debug endpoint: kembalikan semua kandidat kontur yang memenuhi syarat
    deteksi kartu, beserta alasan gagal jika tidak ada yang cocok.
    Berguna untuk mendiagnosis mengapa kartu ATM/KTP tidak terdeteksi.
    """
    CARD_ASPECT = 85.6 / 54.0
    ASPECT_TOL  = 0.30

    try:
        image_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"success": False, "message": "Gagal mendekode gambar"}

        h_img, w_img = image.shape[:2]
        img_area = h_img * w_img

        gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 30, 100)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges   = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_contours = len(contours)
        fail_area      = 0
        fail_shape     = 0
        fail_aspect    = 0
        candidates     = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_pct = area / img_area * 100

            if area < img_area * 0.004 or area > img_area * 0.35:
                fail_area += 1
                continue

            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) not in (4, 5):
                fail_shape += 1
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw == 0 or bh == 0:
                continue

            aspect = bw / bh
            aspect_err_landscape = abs(aspect - CARD_ASPECT) / CARD_ASPECT
            aspect_err_portrait  = abs(aspect - 1.0 / CARD_ASPECT) / (1.0 / CARD_ASPECT)

            if aspect_err_landscape > ASPECT_TOL and aspect_err_portrait > ASPECT_TOL:
                fail_aspect += 1
                candidates.append({
                    "status": "fail_aspect",
                    "bbox": [int(x), int(y), int(bw), int(bh)],
                    "aspect": round(aspect, 3),
                    "area_pct": round(area_pct, 2),
                    "err_landscape": round(aspect_err_landscape, 3),
                    "note": f"Aspek {aspect:.3f}, perlu {CARD_ASPECT:.3f}±30% atau {1/CARD_ASPECT:.3f}±30%",
                })
                continue

            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            solidity  = area / hull_area if hull_area > 0 else 0.0
            score     = solidity * area / img_area

            if aspect_err_landscape <= ASPECT_TOL:
                long_px = float(bw)
                orient  = "landscape"
            else:
                long_px = float(bh)
                orient  = "portrait"

            candidates.append({
                "status": "PASS",
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "aspect": round(aspect, 3),
                "area_pct": round(area_pct, 2),
                "solidity": round(solidity, 3),
                "long_px": round(long_px, 1),
                "orientation": orient,
                "pixels_per_cm": round(long_px / 8.56, 4),
            })

        passed = [c for c in candidates if c["status"] == "PASS"]

        return {
            "success": True,
            "image_size": f"{w_img}x{h_img}",
            "total_contours": total_contours,
            "fail_area": fail_area,
            "fail_shape": fail_shape,
            "fail_aspect": fail_aspect,
            "passed_count": len(passed),
            "candidates": candidates,
            "diagnosis": (
                "Kartu terdeteksi OK" if passed
                else (
                    "Tidak ada kontur lolos filter area — kartu terlalu kecil/besar di frame"
                    if fail_area > total_contours * 0.8
                    else "Kontur ada tapi tidak berbentuk 4-sisi — kartu tertutup atau foto buram"
                    if fail_shape > fail_aspect
                    else "Kontur 4-sisi ada tapi rasio aspek tidak cocok kartu ATM/KTP — kartu miring, terpotong, atau terhalang"
                )
            ),
        }

    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}