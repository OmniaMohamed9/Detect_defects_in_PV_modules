# backend/app.py
import os
import io
import base64
import cv2
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
from quality_check import run_quality_check
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ----------------- Config -----------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
TMP_FOLDER = "tmp"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
# adjust your cell model path or keep fallback to MODEL_PATH
CELL_DETECTION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best2.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TMP_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# ----------------- Load Models -----------------
model = YOLO(MODEL_PATH)
try:
    if os.path.exists(CELL_DETECTION_MODEL_PATH):
        cell_detection_model = YOLO(CELL_DETECTION_MODEL_PATH)
        print(f"✅ Cell detection model loaded: {CELL_DETECTION_MODEL_PATH}")
    else:
        print(f"❌ Cell detection model not found: {CELL_DETECTION_MODEL_PATH}")
        print("Using defect detection model for cell detection as fallback")
        cell_detection_model = model
except Exception as e:
    print(f"❌ Error loading cell detection model: {e}")
    print("Using defect detection model for cell detection as fallback")
    cell_detection_model = model

# ----------------- Utility functions -----------------


def detect_cells_with_boxes(image_path):
    """
    Detect cells, compute raw brightness per cell (mean gray level) and
    compute relative brightness per your requested formula:
        relative = ((a - b) / a) * 100
    where a = brightness of brightest cell, b = brightness of cell.
    Returns (annotated_output_path, cell_detections_list)
    Each cell dict contains: cell_id, confidence, bbox, raw_brightness, relative_brightness
    """
    try:
        print(f"🔍 Running cell detection using model: {CELL_DETECTION_MODEL_PATH}")
        results = cell_detection_model(image_path, imgsz=640)
        img = cv2.imread(image_path)

        if img is None:
            return None, "Could not load image"

        boxes = results[0].boxes
        cell_detections = []
        raw_brightness_values = []

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes.xyxy):
                # convert to ints
                x1, y1, x2, y2 = map(int, box)
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0

                # Extract ROI and compute raw brightness (mean grayscale)
                roi = img[y1:y2, x1:x2]
                raw_brightness = 0.0
                if roi.size > 0:
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    raw_brightness = float(np.mean(gray))
                raw_brightness_values.append(raw_brightness)

                # Draw rectangle and label onto image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"cell {i+1}: {conf:.2f}"
                cv2.putText(img, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save detection with raw brightness (relative set later)
                cell_detections.append({
                    "cell_id": i + 1,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "raw_brightness": raw_brightness
                })

            # Determine 'a' (max raw brightness) and compute relative brightness for each cell
            a = max(raw_brightness_values) if len(raw_brightness_values) > 0 else 0.0
            if a == 0:
                # avoid division by zero: set relative to 0
                for c in cell_detections:
                    c["relative_brightness"] = 0.0
            else:
                for c in cell_detections:
                    b = c.get("raw_brightness", 0.0)
                    # formula: ((a - b)/a) * 100
                    rel = ((a - b) / a) * 100.0
                    c["relative_brightness"] = round(rel, 2)

            total_cells = len(boxes)
            cv2.putText(img, f"Total Cells Detected: {total_cells}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            print(f"Detected {total_cells} cells in the image")
        else:
            print("No cells detected in the image")
            cv2.putText(img, "No Cells Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Save annotated output
        output_path = os.path.join(OUTPUT_FOLDER, f"cell_detection_{os.path.basename(image_path)}")
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"✅ Cell detection image saved to: {output_path}")
            if os.path.exists(output_path):
                print(f"✅ File verified. Size: {os.path.getsize(output_path)} bytes")
        else:
            print(f"❌ Failed to save cell detection image to: {output_path}")

        # Logging brightness info
        print("Raw & Relative brightness per cell:")
        for c in cell_detections:
            print(f"Cell {c['cell_id']} -> raw: {c['raw_brightness']:.2f}, relative: {c['relative_brightness']}%")

        return output_path, cell_detections

    except Exception as e:
        print(f"Error in cell detection: {e}")
        return None, str(e)


def calculate_coordinate_overlap(box1, box2, threshold=0.5):
    """Return (overlaps_bool, iou_value) comparing two [x1,y1,x2,y2] boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return False, 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = max(0, (x2_1 - x1_1) * (y2_1 - y1_1))
    area2 = max(0, (x2_2 - x1_2) * (y2_2 - y1_2))
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou > threshold, iou


def map_defects_to_cells(defect_predictions, cell_detections, img_width, img_height):
    """Map defect predictions (with absolute pixel coords) to detected cells by IoU."""
    mapped_results = []

    for defect in defect_predictions:
        defect_bbox = defect["xyxy"]  # [x1,y1,x2,y2]
        defect_label = defect["label"]
        defect_confidence = defect["confidence"]

        best_match = None
        best_iou = 0.0

        for cell in cell_detections:
            cell_bbox = cell["bbox"]
            overlaps, iou = calculate_coordinate_overlap(defect_bbox, cell_bbox, threshold=0.3)
            if overlaps and iou > best_iou:
                best_match = cell
                best_iou = iou

        if best_match:
            mapped_results.append({
                "cell_id": best_match["cell_id"],
                "cell_confidence": best_match.get("confidence", 0.0),
                "defect_type": defect_label,
                "defect_confidence": defect_confidence,
                "overlap_iou": best_iou,
                "defect_bbox": defect_bbox,
                "cell_bbox": best_match["bbox"]
            })
        else:
            mapped_results.append({
                "cell_id": "Unknown",
                "cell_confidence": 0.0,
                "defect_type": defect_label,
                "defect_confidence": defect_confidence,
                "overlap_iou": 0.0,
                "defect_bbox": defect_bbox,
                "cell_bbox": None
            })

    return mapped_results


def calculate_darkness_percentage(image_path, x1, y1, x2, y2, img_width, img_height):
    """Calculate darkness percentage in defect region (existing logic preserved)."""
    try:
        img = cv2.imread(image_path)
        h, w, _ = img.shape

        x1_actual = int((x1 / img_width) * w)
        y1_actual = int((y1 / img_height) * h)
        x2_actual = int((x2 / img_width) * w)
        y2_actual = int((y2 / img_height) * h)

        x1_actual = max(0, min(x1_actual, w - 1))
        y1_actual = max(0, min(y1_actual, h - 1))
        x2_actual = max(x1_actual + 1, min(x2_actual, w))
        y2_actual = max(y1_actual + 1, min(y2_actual, h))

        roi = img[y1_actual:y2_actual, x1_actual:x2_actual]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        clean_intensities = []
        for _ in range(10):
            sample_x = np.random.randint(0, max(1, w - 50))
            sample_y = np.random.randint(0, max(1, h - 50))
            if not (sample_x < x2_actual and sample_x + 50 > x1_actual and
                    sample_y < y2_actual and sample_y + 50 > y1_actual):
                sample_roi = img[sample_y:sample_y + 50, sample_x:sample_x + 50]
                if sample_roi.size > 0:
                    sample_gray = cv2.cvtColor(sample_roi, cv2.COLOR_BGR2GRAY)
                    clean_intensities.append(np.mean(sample_gray))

        if len(clean_intensities) > 0:
            clean_ref = np.mean(clean_intensities)
            darkness = max(0, min(100, ((clean_ref - mean_intensity) / clean_ref) * 100))
        else:
            darkness = max(0, min(100, (255 - mean_intensity) / 255 * 100))

        return round(darkness, 2)
    except Exception as e:
        print(f"Error calculating darkness: {e}")
        return 0.0


# ----------------- Routes -----------------


@app.route("/detect", methods=["POST"])
def detect():
    # require three images: image1 (EL1), image2 (EL2), image3 (ELBG)
    if "image1" not in request.files or "image2" not in request.files or "image3" not in request.files:
        return jsonify({"success": False, "error": "All three images (EL1, EL2, ELBG) are required"}), 400

    el1_file = request.files["image1"]
    el2_file = request.files["image2"]
    elbg_file = request.files["image3"]

    if el1_file.filename == "" or el2_file.filename == "" or elbg_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    el1_fname = secure_filename(el1_file.filename)
    el2_fname = secure_filename(el2_file.filename)
    elbg_fname = secure_filename(elbg_file.filename)

    el1_path = os.path.join(UPLOAD_FOLDER, el1_fname)
    el2_path = os.path.join(UPLOAD_FOLDER, el2_fname)
    elbg_path = os.path.join(UPLOAD_FOLDER, elbg_fname)

    el1_file.save(el1_path)
    el2_file.save(el2_path)
    elbg_file.save(elbg_path)

    # Quality check
    print("\n🔍 Running quality check before YOLO detection...")
    try:
        quality_result = run_quality_check(el1_path, el2_path, elbg_path)
        if not quality_result.get("accepted", False):
            print("❌ Quality check failed!")
        else:
            print("✅ Quality check passed!")
    except Exception as e:
        print(f"❌ Quality check error: {e}")
        return jsonify({"success": False, "error": f"Quality check failed: {str(e)}"}), 500

    # Cell detection
    print("\n🔍 Running cell detection on EL1 image...")
    cell_detection_path, cell_detections = detect_cells_with_boxes(el1_path)
    if cell_detection_path is None:
        return jsonify({"success": False, "error": f"Cell detection failed: {cell_detections}"}), 500

    # encode cell detection image as base64
    try:
        if os.path.exists(cell_detection_path):
            with open(cell_detection_path, "rb") as f:
                cell_detection_b64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            cell_detection_b64 = ""
    except Exception as e:
        print(f"Error encoding cell detection image: {e}")
        cell_detection_b64 = ""

    # Run defect detection on EL1
    model_results = model(el1_path, imgsz=640)
    r = model_results[0]

    # annotated image as base64
    try:
        annotated = r.plot()
        pil_img = Image.fromarray(annotated)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        annotated_b64 = ""

    preds = []
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        xyxy_all = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else []
        conf_all = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else []
        cls_all = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") else []
        for i in range(len(xyxy_all)):
            x1, y1, x2, y2 = xyxy_all[i].tolist()
            conf = float(conf_all[i]) if len(conf_all) > 0 else 0.0
            cls = int(cls_all[i]) if len(cls_all) > 0 else -1
            label = r.names[cls] if (hasattr(r, "names") and cls in r.names) else str(cls)
            preds.append({"label": label, "confidence": conf, "xyxy": [x1, y1, x2, y2]})

    # Map defects to cells
    print("\n🔍 Mapping defects to cells based on coordinate overlap...")
    try:
        img = Image.open(el1_path)
        img_width, img_height = img.size
        mapped_defects = map_defects_to_cells(preds, cell_detections, img_width, img_height)
        print(f"✅ Mapped {len(mapped_defects)} defects to cells")
    except Exception as e:
        print(f"❌ Error mapping defects to cells: {e}")
        mapped_defects = []

    # Logging cell detections
    print("Returning cell detections:")
    for c in cell_detections:
        print(c)

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj

    # Convert all data to JSON-serializable format
    cell_detections_serializable = convert_numpy_types(cell_detections)
    mapped_defects_serializable = convert_numpy_types(mapped_defects)
    preds_serializable = convert_numpy_types(preds)
    quality_result_serializable = convert_numpy_types(quality_result)

    # Build response
    return jsonify({
        "success": True,
        "quality_result": quality_result_serializable,
        "cell_detection": {
            "image": cell_detection_b64,
            "detections": cell_detections_serializable,
            "total_cells": len(cell_detections_serializable)
        },
        "cell_detection_image": cell_detection_b64,
        "cell_detections": cell_detections_serializable,
        "defect_mapping": mapped_defects_serializable,
        "defect_results": mapped_defects_serializable,
        "results": [{
            "image_name": el1_fname,
            "annotated_image": annotated_b64,
            "predictions": preds_serializable
        }],
        "predictions": preds_serializable
    })


@app.route("/analyze_defects", methods=["POST"])
def analyze_defects():
    """Analyze single image for defects and return grid cell mapping"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Image is required"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    try:
        results = model(image_path, conf=0.5)
        r = results[0]
        img = Image.open(image_path)
        img_width, img_height = img.size

        detections = []
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            xyxy_all = r.boxes.xyxy.cpu().numpy()
            conf_all = r.boxes.conf.cpu().numpy()
            cls_all = r.boxes.cls.cpu().numpy()

            for i in range(len(xyxy_all)):
                x1, y1, x2, y2 = xyxy_all[i].tolist()
                conf = float(conf_all[i])
                cls = int(cls_all[i])
                label = r.names[cls] if hasattr(r, "names") and cls in r.names else str(cls)

                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height

                darkness = calculate_darkness_percentage(image_path, x1, y1, x2, y2, img_width, img_height)

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "x1": x1_norm,
                    "y1": y1_norm,
                    "x2": x2_norm,
                    "y2": y2_norm,
                    "darkness": darkness
                })

        return jsonify({
            "success": True,
            "detections": detections,
            "image_dimensions": {"width": img_width, "height": img_height}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf_route():
    """Generate a PDF report from posted JSON data.
       Expected data fields: cell_detection_image (base64), results (annotated), quality_result, cell_detections, rows, columns, camera_model, location, notes, brightness_levels (optional)
    """
    try:
        data = request.get_json()
        if not data or "cell_detection_image" not in data:
            return jsonify({"success": False, "error": "Cell detection image data is required"}), 400

        pdf_filename = f"analysis_report_{int(time.time())}.pdf"
        pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12,
            alignment=1
        )
        story.append(Paragraph("Solar Panel Analysis Report", title_style))
        story.append(Spacer(1, 12))

        # Camera + Quality side-by-side
        left_info = [
            Paragraph(f"Camera Model: {data.get('camera_model','N/A')}", styles['Normal']),
            Paragraph(f"Location: {data.get('location','N/A')}", styles['Normal']),
            Paragraph(f"Notes: {data.get('notes','N/A')}", styles['Normal'])
        ]
        right_info = []
        if "quality_result" in data:
            q = data["quality_result"]
            status = "❌ NOT PASSED"
            snr_value = q.get("snr50", None)
            if snr_value is not None and snr_value > 45:
                status = "✅ PASSED"
            right_info.append(Paragraph(f"Quality Check: {status}", styles['Normal']))
            if snr_value is not None:
                right_info.append(Paragraph(f"SNR50: {snr_value:.2f}", styles['Normal']))
            if "sharpness_mm" in q:
                right_info.append(Paragraph(f"Sharpness (mm): {q['sharpness_mm']:.2f}", styles['Normal']))
            if "sharpness_category" in q:
                right_info.append(Paragraph(f"Sharpness Category: {q['sharpness_category']}", styles['Normal']))
            if "details" in q:
                d = q["details"]
                if "histogram_mean" in d:
                    right_info.append(Paragraph(f"Histogram Mean: {d['histogram_mean']:.2f}", styles['Normal']))
                if "histogram_variance" in d:
                    right_info.append(Paragraph(f"Histogram Variance: {d['histogram_variance']:.2f}", styles['Normal']))

        table = Table([[left_info, right_info]], colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

        # Cell detection image
        img_b64 = data.get("cell_detection_image", "")
        if img_b64:
            tmp_img_path = os.path.join(TMP_FOLDER, f"cell_{int(time.time())}.jpg")
            with open(tmp_img_path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            story.append(Paragraph("Cell Detection Image", styles['Heading2']))
            story.append(RLImage(tmp_img_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 12))

        # Defect annotated image
        if "results" in data and len(data["results"]) > 0:
            annotated_b64 = data["results"][0].get("annotated_image", "")
            if annotated_b64:
                tmp_def_path = os.path.join(TMP_FOLDER, f"defect_{int(time.time())}.jpg")
                with open(tmp_def_path, "wb") as f:
                    f.write(base64.b64decode(annotated_b64))
                story.append(Paragraph("Defect Detection Image", styles['Heading2']))
                story.append(RLImage(tmp_def_path, width=6*inch, height=4*inch))
                story.append(Spacer(1, 12))

        # Grid table
        rows = int(data.get("rows", 0) or 0)
        cols = int(data.get("columns", 0) or 0)
        if rows > 0 and cols > 0:
            grid = [[f"{r},{chr(64 + c)}" for c in range(1, cols + 1)] for r in range(1, rows + 1)]
            story.append(Paragraph("Cell Grid Layout", styles['Heading2']))
            t = Table(grid, colWidths=[0.6*inch] * cols)
            t.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 12))

        # Cells Data table (raw brightness + relative brightness)
        if "cell_detections" in data and len(data["cell_detections"]) > 0:
            story.append(Paragraph("Cell Brightness (Raw and Relative)", styles['Heading2']))

            # Table header
            cell_table_data = [["Cell ID", "Raw Brightness", "Relative Brightness (%)", "Confidence"]]

            # find brightest cell to optionally highlight (lowest relative brightness is the brightest: relative==0)
            brightest_cell_id = None
            min_rel = None
            for c in data["cell_detections"]:
                rel = c.get("relative_brightness", None)
                if rel is not None:
                    if min_rel is None or rel < min_rel:
                        min_rel = rel
                        brightest_cell_id = c.get("cell_id")

            for c in data["cell_detections"]:
                cell_table_data.append([
                    str(c.get("cell_id", "Unknown")),
                    f"{c.get('raw_brightness', 0):.2f}",
                    f"{c.get('relative_brightness', 0):.2f}",
                    f"{c.get('confidence', 0):.2f}"
                ])

            ctable = Table(cell_table_data, colWidths=[1*inch, 1.5*inch, 1.8*inch, 1*inch])
            # style: highlight brightest row (if found)
            table_style = [
                ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
            ]
            # find which row index corresponds to brightest_cell_id (header row is 0)
            if brightest_cell_id is not None:
                for row_idx in range(1, len(cell_table_data)):
                    try:
                        if int(cell_table_data[row_idx][0]) == int(brightest_cell_id):
                            # highlight this row light green
                            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), colors.lavender))
                            break
                    except Exception:
                        pass

            ctable.setStyle(TableStyle(table_style))
            story.append(ctable)
            story.append(Spacer(1, 12))

        # Brightness Classification (optional thresholds)
        brightness_levels = sorted(data.get("brightness_levels", []))
        if brightness_levels and "cell_detections" in data and len(data["cell_detections"]) > 0:
            story.append(Paragraph("Brightness Classification", styles['Heading2']))
            counts = [0] * (len(brightness_levels) + 1)
            for c in data["cell_detections"]:
                b = c.get('relative_brightness', 0)
                placed = False
                for idx, th in enumerate(brightness_levels):
                    if b < th:
                        counts[idx] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1
            for i, th in enumerate(brightness_levels):
                story.append(Paragraph(f"Cells with relative brightness < {th}: {counts[i]}", styles['Normal']))
            story.append(Paragraph(f"Cells with relative brightness ≥ {brightness_levels[-1]}: {counts[-1]}", styles['Normal']))
            story.append(Spacer(1, 12))

        # Mapped defects listing
        if "defect_mapping" in data and len(data["defect_mapping"]) > 0:
            story.append(Paragraph("Mapped Defects to Cells", styles['Heading2']))
            for i, m in enumerate(data["defect_mapping"]):
                story.append(Paragraph(
                    f"Defect {i + 1}: Cell {m.get('cell_id','Unknown')} — "
                    f"{m.get('defect_type','Unknown')} "
                    f"(Conf: {m.get('defect_confidence',0):.2f}, IoU: {m.get('overlap_iou',0):.2f})",
                    styles['Normal']
                ))
            story.append(Spacer(1, 12))

        # Build PDF
        doc.build(story)

        # return PDF as base64 for API convenience
        with open(pdf_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({"success": True, "pdf_filename": os.path.basename(pdf_path), "pdf_data": pdf_b64})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ----------------- Main -----------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
