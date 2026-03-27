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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

UPLOAD_FOLDER = "uploads"
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
CELL_DETECTION_MODEL_PATH = r"C:\Users\DELL\Downloads\demo\demo\demo\best2.pt"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# Load YOLO models
model = YOLO(MODEL_PATH)  # For defect detection

# Load cell detection model with error handling
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

def detect_cells_with_boxes(image_path):
    """Detect cells using YOLO and draw bounding boxes around them"""
    try:
        # Run YOLO inference using the cell detection model
        print(f"🔍 Running cell detection using model: {CELL_DETECTION_MODEL_PATH}")
        results = cell_detection_model(image_path, imgsz=640)
        img = cv2.imread(image_path)
        
        if img is None:
            return None, "Could not load image"
        
        # Get detection results
        boxes = results[0].boxes
        cell_detections = []
        
        if boxes is not None and len(boxes) > 0:
            # Draw boxes around each detected cell
            for i, box in enumerate(boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                
                # Get confidence score for this specific box
                conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                
                # Draw rectangle around the cell
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence score
                label = f"cell {i+1}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Store detection info
                cell_detections.append({
                    "cell_id": i+1,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
            
            # Add total count information
            total_cells = len(boxes)
            cv2.putText(img, f"Total Cells Detected: {total_cells}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            print(f"Detected {total_cells} cells in the image")
        else:
            print("No cells detected in the image")
            cv2.putText(img, "No Cells Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save the annotated image
        output_path = os.path.join("outputs", f"cell_detection_{os.path.basename(image_path)}")
        os.makedirs("outputs", exist_ok=True)
        
        # Ensure the image is saved correctly
        success = cv2.imwrite(output_path, img)
        if success:
            print(f"✅ Cell detection image saved to: {output_path}")
            # Verify file exists and has content
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✅ File verified. Size: {file_size} bytes")
            else:
                print(f"❌ File was not created: {output_path}")
        else:
            print(f"❌ Failed to save cell detection image to: {output_path}")
        
        return output_path, cell_detections
        
    except Exception as e:
        print(f"Error in cell detection: {e}")
        return None, str(e)

def calculate_coordinate_overlap(box1, box2, threshold=0.5):
    """Calculate overlap between two bounding boxes and return True if overlap > threshold"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return False, 0.0  # No intersection
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU (Intersection over Union)
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou > threshold, iou

def map_defects_to_cells(defect_predictions, cell_detections, img_width, img_height):
    """Map defect predictions to specific cells based on coordinate overlap"""
    mapped_results = []
    
    for defect in defect_predictions:
        defect_bbox = defect["xyxy"]  # [x1, y1, x2, y2]
        defect_label = defect["label"]
        defect_confidence = defect["confidence"]
        
        # Find the best matching cell
        best_match = None
        best_iou = 0.0
        
        for cell in cell_detections:
            cell_bbox = cell["bbox"]  # [x1, y1, x2, y2]
            
            # Check if defect overlaps with this cell
            overlaps, iou = calculate_coordinate_overlap(defect_bbox, cell_bbox, threshold=0.3)
            
            if overlaps and iou > best_iou:
                best_match = cell
                best_iou = iou
        
        if best_match:
            mapped_results.append({
                "cell_id": best_match["cell_id"],
                "cell_confidence": best_match["confidence"],
                "defect_type": defect_label,
                "defect_confidence": defect_confidence,
                "overlap_iou": best_iou,
                "defect_bbox": defect_bbox,
                "cell_bbox": best_match["bbox"]
            })
        else:
            # Defect doesn't overlap with any detected cell
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
    """Calculate darkness percentage for a detected defect region"""
    try:
        # Load image with OpenCV
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        
        # Convert pixel coordinates to actual image coordinates
        x1_actual = int((x1 / img_width) * w)
        y1_actual = int((y1 / img_height) * h)
        x2_actual = int((x2 / img_width) * w)
        y2_actual = int((y2 / img_height) * h)
        
        # Ensure coordinates are within image bounds
        x1_actual = max(0, min(x1_actual, w-1))
        y1_actual = max(0, min(y1_actual, h-1))
        x2_actual = max(x1_actual+1, min(x2_actual, w))
        y2_actual = max(y1_actual+1, min(y2_actual, h))
        
        # Extract ROI
        roi = img[y1_actual:y2_actual, x1_actual:x2_actual]
        if roi.size == 0:
            return 0.0
            
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        
        # Calculate reference intensity from clean areas
        # Sample some random areas outside the defect region
        clean_intensities = []
        for _ in range(10):
            # Random sample outside the defect area
            sample_x = np.random.randint(0, w - 50)
            sample_y = np.random.randint(0, h - 50)
            
            # Check if sample is not overlapping with defect
            if not (sample_x < x2_actual and sample_x + 50 > x1_actual and 
                    sample_y < y2_actual and sample_y + 50 > y1_actual):
                sample_roi = img[sample_y:sample_y+50, sample_x:sample_x+50]
                if sample_roi.size > 0:
                    sample_gray = cv2.cvtColor(sample_roi, cv2.COLOR_BGR2GRAY)
                    clean_intensities.append(np.mean(sample_gray))
        
        if len(clean_intensities) > 0:
            clean_ref = np.mean(clean_intensities)
            darkness = max(0, min(100, ((clean_ref - mean_intensity) / clean_ref) * 100))
        else:
            # Fallback calculation
            darkness = max(0, min(100, (255 - mean_intensity) / 255 * 100))
            
        return round(darkness, 2)
        
    except Exception as e:
        print(f"Error calculating darkness: {e}")
        return 0.0

@app.route("/detect", methods=["POST"])
def detect():
    # Check for all three required images
    if "image1" not in request.files or "image2" not in request.files or "image3" not in request.files:
        return jsonify({"success": False, "error": "All three images (EL1, EL2, ELBG) are required"}), 400

    # Get the three images
    el1_file = request.files["image1"]
    el2_file = request.files["image2"] 
    elbg_file = request.files["image3"]

    if el1_file.filename == "" or el2_file.filename == "" or elbg_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # Save all three images
    el1_fname = secure_filename(el1_file.filename)
    el2_fname = secure_filename(el2_file.filename)
    elbg_fname = secure_filename(elbg_file.filename)
    
    el1_path = os.path.join(UPLOAD_FOLDER, el1_fname)
    el2_path = os.path.join(UPLOAD_FOLDER, el2_fname)
    elbg_path = os.path.join(UPLOAD_FOLDER, elbg_fname)
    
    el1_file.save(el1_path)
    el2_file.save(el2_path)
    elbg_file.save(elbg_path)

    # Run quality check first
    print("\n🔍 Running quality check before YOLO detection...")
    try:
        quality_result = run_quality_check(el1_path, el2_path, elbg_path)
        
        # Check if quality is acceptable
        if not quality_result["accepted"]:
            return jsonify({
                "success": False, 
                "error": "Image quality check failed",
                "quality_result": quality_result
            }), 400
            
        print("✅ Quality check passed! Proceeding with YOLO detection...")
        
    except Exception as e:
        print(f"❌ Quality check error: {e}")
        return jsonify({"success": False, "error": f"Quality check failed: {str(e)}"}), 500

    # Run cell detection on the first image (EL1)
    print("\n🔍 Running cell detection on EL1 image...")
    cell_detection_path, cell_detections = detect_cells_with_boxes(el1_path)
    
    if cell_detection_path is None:
        return jsonify({"success": False, "error": f"Cell detection failed: {cell_detections}"}), 500
    
    # Convert cell detection image to base64
    try:
        print(f"🔍 Converting cell detection image to base64: {cell_detection_path}")
        if os.path.exists(cell_detection_path):
            with open(cell_detection_path, "rb") as img_file:
                img_data = img_file.read()
                cell_detection_b64 = base64.b64encode(img_data).decode("utf-8")
                print(f"✅ Cell detection image encoded to base64. Size: {len(cell_detection_b64)} characters")
        else:
            print(f"❌ Cell detection image file does not exist: {cell_detection_path}")
            cell_detection_b64 = ""
    except Exception as e:
        print(f"❌ Error encoding cell detection image: {e}")
        cell_detection_b64 = ""

    # Run inference on EL1 for defect detection (after quality check passes)
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

    # Map defects to cells based on coordinate overlap
    print("\n🔍 Mapping defects to cells based on coordinate overlap...")
    try:
        # Get image dimensions for mapping
        img = Image.open(el1_path)
        img_width, img_height = img.size
        
        # Map defects to cells
        mapped_defects = map_defects_to_cells(preds, cell_detections, img_width, img_height)
        print(f"✅ Mapped {len(mapped_defects)} defects to cells")
        
        # Log mapping results
        for mapping in mapped_defects:
            if mapping["cell_id"] != "Unknown":
                print(f"  Cell {mapping['cell_id']}: {mapping['defect_type']} (IoU: {mapping['overlap_iou']:.3f})")
            else:
                print(f"  Unknown cell: {mapping['defect_type']} (no overlap with detected cells)")
                
    except Exception as e:
        print(f"❌ Error mapping defects to cells: {e}")
        mapped_defects = []

    return jsonify({
        "success": True,
        "quality_result": quality_result,
        "cell_detection": {
            "image": cell_detection_b64,
            "detections": cell_detections,
            "total_cells": len(cell_detections)
        },
        "defect_mapping": mapped_defects,
        "results": [{
            "image_name": el1_fname,
            "annotated_image": annotated_b64,
            "predictions": preds
        }]
    })

@app.route("/analyze_defects", methods=["POST"])
def analyze_defects():
    """Analyze single image for defects and return grid cell mapping"""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Image is required"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    # Save image
    filename = secure_filename(image_file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    try:
        # Run YOLO detection
        results = model(image_path, conf=0.5)
        r = results[0]
        
        # Get image dimensions
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
                
                # Normalize coordinates to 0-1 range
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                
                # Calculate darkness percentage based on intensity analysis
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
def generate_pdf():
    """Generate PDF report with cell detection and defect analysis"""
    try:
        data = request.get_json()
        
        if not data or "cell_detection_image" not in data:
            return jsonify({"success": False, "error": "Cell detection image data is required"}), 400
        
        # Create PDF filename
        pdf_filename = f"analysis_report_{int(time.time())}.pdf"
        pdf_path = os.path.join("outputs", pdf_filename)
        os.makedirs("outputs", exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Solar Panel Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Cell Detection Section
        story.append(Paragraph("Cell Detection Results", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Add cell detection image
        temp_img_path = None
        try:
            # Decode base64 image
            cell_detection_b64 = data["cell_detection_image"]
            if cell_detection_b64:
                # Save temporary image
                temp_img_path = os.path.join("tmp", f"temp_cell_detection_{int(time.time())}.jpg")
                os.makedirs("tmp", exist_ok=True)
                
                print(f"🔍 Saving cell detection image to: {temp_img_path}")
                with open(temp_img_path, "wb") as img_file:
                    img_file.write(base64.b64decode(cell_detection_b64))
                
                # Verify file was created
                if os.path.exists(temp_img_path):
                    file_size = os.path.getsize(temp_img_path)
                    print(f"✅ Cell detection image saved successfully. Size: {file_size} bytes")
                    
                    # Add image to PDF
                    try:
                        # Try to create ReportLab Image with error handling
                        img = RLImage(temp_img_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 12))
                        print("✅ Cell detection image added to PDF story")
                    except Exception as img_error:
                        print(f"❌ Error adding image to PDF: {img_error}")
                        # Try alternative approach - convert to PIL and save as different format
                        try:
                            from PIL import Image as PILImage
                            pil_img = PILImage.open(temp_img_path)
                            # Convert to RGB if necessary
                            if pil_img.mode != 'RGB':
                                pil_img = pil_img.convert('RGB')
                            # Save as PNG for better compatibility
                            png_path = temp_img_path.replace('.jpg', '.png')
                            pil_img.save(png_path, 'PNG')
                            
                            # Try again with PNG
                            img = RLImage(png_path, width=6*inch, height=4*inch)
                            story.append(img)
                            story.append(Spacer(1, 12))
                            print("✅ Cell detection image added to PDF story (converted to PNG)")
                            
                            # Clean up PNG file
                            if os.path.exists(png_path):
                                os.remove(png_path)
                                
                        except Exception as convert_error:
                            print(f"❌ Error converting image: {convert_error}")
                            story.append(Paragraph("Error: Could not add cell detection image to PDF", styles['Normal']))
                else:
                    print("❌ Cell detection image file was not created")
                    story.append(Paragraph("Error: Cell detection image could not be saved", styles['Normal']))
                
                # Add cell detection summary
                if "cell_detections" in data:
                    total_cells = len(data["cell_detections"])
                    story.append(Paragraph(f"Total Cells Detected: {total_cells}", styles['Normal']))
                    
                    if total_cells > 0:
                        story.append(Paragraph("Cell Analysis:", styles['Heading3']))
                        
                        # Create detailed cell analysis with defect mapping
                        if "defect_mapping" in data and data["defect_mapping"]:
                            story.append(Paragraph("Defect-to-Cell Mapping Results:", styles['Heading3']))
                            
                            # Group defects by cell
                            cell_defect_map = {}
                            unknown_defects = []
                            
                            for mapping in data["defect_mapping"]:
                                cell_id = mapping.get("cell_id", "Unknown")
                                if cell_id == "Unknown":
                                    unknown_defects.append(mapping)
                                else:
                                    if cell_id not in cell_defect_map:
                                        cell_defect_map[cell_id] = []
                                    cell_defect_map[cell_id].append(mapping)
                            
                            # Display mapped defects
                            for cell_id in sorted(cell_defect_map.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
                                defects = cell_defect_map[cell_id]
                                story.append(Paragraph(f"Cell {cell_id}:", styles['Heading4']))
                                
                                for defect in defects:
                                    defect_type = defect.get("defect_type", "Unknown")
                                    defect_conf = defect.get("defect_confidence", 0)
                                    overlap_iou = defect.get("overlap_iou", 0)
                                    
                                    story.append(Paragraph(f"  • {defect_type} (Confidence: {defect_conf:.3f}, IoU: {overlap_iou:.3f})", 
                                                         styles['Normal']))
                            
                            # Display unknown defects
                            if unknown_defects:
                                story.append(Paragraph("Defects not mapped to any cell:", styles['Heading4']))
                                for defect in unknown_defects:
                                    defect_type = defect.get("defect_type", "Unknown")
                                    defect_conf = defect.get("defect_confidence", 0)
                                    story.append(Paragraph(f"  • {defect_type} (Confidence: {defect_conf:.3f})", 
                                                         styles['Normal']))
                        
                        # Also show overall cell health status
                        story.append(Paragraph("Cell Health Summary:", styles['Heading3']))
                        healthy_cells = []
                        defective_cells = []
                        
                        for i, cell in enumerate(data["cell_detections"]):
                            cell_id = cell.get("cell_id", i+1)
                            conf = cell.get("confidence", 0)
                            
                            # Check if this cell has any defects
                            has_defects = False
                            if "defect_mapping" in data:
                                for mapping in data["defect_mapping"]:
                                    if mapping.get("cell_id") == cell_id:
                                        has_defects = True
                                        break
                            
                            if has_defects:
                                defective_cells.append(cell_id)
                            else:
                                healthy_cells.append(cell_id)
                        
                        story.append(Paragraph(f"Healthy Cells: {len(healthy_cells)} - {', '.join(map(str, healthy_cells[:10]))}", 
                                             styles['Normal']))
                        if len(healthy_cells) > 10:
                            story.append(Paragraph(f"... and {len(healthy_cells) - 10} more healthy cells", styles['Normal']))
                        
                        story.append(Paragraph(f"Defective Cells: {len(defective_cells)} - {', '.join(map(str, defective_cells))}", 
                                             styles['Normal']))
                        
                        if total_cells > 10:
                            story.append(Paragraph(f"... and {total_cells - 10} more cells", styles['Normal']))
            else:
                print("❌ No cell detection image data provided")
                story.append(Paragraph("No cell detection image available", styles['Normal']))
                    
        except Exception as e:
            print(f"❌ Error processing cell detection image: {str(e)}")
            story.append(Paragraph(f"Error processing cell detection image: {str(e)}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Defect Analysis Section
        if "defect_results" in data:
            story.append(Paragraph("Defect Analysis Results", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            defect_results = data["defect_results"]
            if defect_results and len(defect_results) > 0:
                story.append(Paragraph(f"Defects Found: {len(defect_results)}", styles['Normal']))
                
                for i, defect in enumerate(defect_results):
                    label = defect.get("label", "Unknown")
                    confidence = defect.get("confidence", 0)
                    story.append(Paragraph(f"Defect {i+1}: {label} (Confidence: {confidence:.2f})", styles['Normal']))
            else:
                story.append(Paragraph("No defects detected", styles['Normal']))
        
        # Quality Check Section
        if "quality_result" in data:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Quality Check Results", styles['Heading2']))
            story.append(Spacer(1, 12))
            
            quality = data["quality_result"]
            status = "PASSED" if quality.get("accepted", False) else "FAILED"
            story.append(Paragraph(f"Quality Check: {status}", styles['Normal']))
            
            if "details" in quality:
                for key, value in quality["details"].items():
                    story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        
        # Build PDF
        print("🔍 Building PDF document...")
        doc.build(story)
        print("✅ PDF document built successfully")
        
        # Clean up temporary image file after PDF is built
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
                print(f"✅ Cleaned up temporary image: {temp_img_path}")
            except Exception as e:
                print(f"⚠️ Could not clean up temporary image: {e}")
        
        # Return PDF as base64
        with open(pdf_path, "rb") as pdf_file:
            pdf_b64 = base64.b64encode(pdf_file.read()).decode("utf-8")
        
        return jsonify({
            "success": True,
            "pdf_filename": pdf_filename,
            "pdf_data": pdf_b64
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
