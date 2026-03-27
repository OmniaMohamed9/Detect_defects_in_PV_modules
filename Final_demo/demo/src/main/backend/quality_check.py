# backend/quality_check.py
import numpy as np
from PIL import Image
import cv2
import os
from scipy.stats import skew, kurtosis

def load_image_as_array(image_path):
    """Load image and convert to numpy array"""
    img = Image.open(image_path)
    return np.array(img)

def create_vcut_mask(image, threshold=30):
    """Create vertical cut mask for image analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return mask

def calculate_snr(el1, el2, elbg, mask=None):
    """Calculate Signal-to-Noise Ratio"""
    if mask is not None:
        el1_masked = el1[mask > 0]
        el2_masked = el2[mask > 0]
        elbg_masked = elbg[mask > 0]
    else:
        el1_masked = el1.flatten()
        el2_masked = el2.flatten()
        elbg_masked = elbg.flatten()
    
    # Calculate SNR
    signal = np.mean(el1_masked) - np.mean(elbg_masked)
    noise = np.std(el2_masked - el1_masked)
    snr = signal / noise if noise > 0 else 0
    
    # Calculate el_final
    el_final = el1 - elbg
    
    return snr * 50, el_final  # Scale SNR to 0-100 range

def calculate_sharpness(image, mask=None):
    """Calculate image sharpness using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    if mask is not None:
        gray = gray[mask > 0]
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def classify_sharpness_mm(sharpness_val):
    """Classify sharpness and convert to mm scale"""
    if sharpness_val > 1000:
        return 1.0, "Excellent"
    elif sharpness_val > 500:
        return 2.5, "Good"
    elif sharpness_val > 200:
        return 5.0, "Fair"
    else:
        return 10.0, "Poor"

def histogram_analysis(image, mask=None):
    """Analyze image histogram statistics"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    if mask is not None:
        gray = gray[mask > 0]
    
    mean_val = np.mean(gray)
    var_val = np.var(gray)
    skew_val = skew(gray.flatten())
    kurt_val = kurtosis(gray.flatten())
    
    return mean_val, var_val, skew_val, kurt_val

def run_quality_check(el1_path, el2_path, elbg_path):
    from pathlib import Path
    
    print("=" * 60)
    print("🔍 Starting image quality check...")
    print("=" * 60)
    
    # Load images
    EL1 = load_image_as_array(el1_path)
    EL2 = load_image_as_array(el2_path)
    ELBG = load_image_as_array(elbg_path)
    
    print(f"📸 Images loaded:")
    print(f"   - First image (EL1): {os.path.basename(el1_path)}")
    print(f"   - Second image (EL2): {os.path.basename(el2_path)}")
    print(f"   - Background image (ELBG): {os.path.basename(elbg_path)}")

    # Generate mask
    vcut_mask = create_vcut_mask(EL1, threshold=30)

    # Calculate metrics
    snr50, el_final = calculate_snr(EL1, EL2, ELBG, mask=vcut_mask)
    el_final_norm = cv2.normalize(el_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sharpness_val = calculate_sharpness(el_final_norm, mask=vcut_mask)
    sharpness_mm, sharpness_category = classify_sharpness_mm(sharpness_val)

    mean_val, var_val, skew_val, kurt_val = histogram_analysis(el_final_norm, mask=vcut_mask)

    # Print SNR and Sharpness in terminal
    print("\n📊 Quality check results:")
    print("-" * 40)
    print(f"🎯 SNR (Signal-to-Noise Ratio): {snr50:.2f}")
    print(f"🔍 Sharpness Value: {sharpness_val:.2f}")
    print(f"📏 Sharpness (mm): {sharpness_mm:.1f} mm")
    print(f"⭐ Sharpness Category: {sharpness_category}")
    print("-" * 40)
    
    # Print histogram statistics
    print("📈 Histogram statistics:")
    print(f"   - Mean: {mean_val:.2f}")
    print(f"   - Variance: {var_val:.2f}")
    print(f"   - Skewness: {skew_val:.2f}")
    print(f"   - Kurtosis: {kurt_val:.2f}")

    # Accept/Reject
    accepted = snr50 >= 45 and sharpness_mm <= 10.0
    
    print(f"\n✅ Check result: {'Accepted' if accepted else 'Rejected'}")
    if not accepted:
        if snr50 < 30:
            print(f"   ❌ SNR too low: {snr50:.2f} (Required: ≥30)")
        if sharpness_mm > 10.0:
            print(f"   ❌ Poor sharpness: {sharpness_mm:.1f} mm (Required: ≤5.0 mm)")
    
    print("=" * 60)

    return {
        "snr50": float(snr50),
        "sharpness_mm": float(sharpness_mm),
        "sharpness_category": sharpness_category,
        "histogram": {
            "mean": float(mean_val),
            "variance": float(var_val),
            "skewness": float(skew_val),
            "kurtosis": float(kurt_val)
        },
        "accepted": accepted
    }

def test_with_sample_images():
    """Test SNR and Sharpness calculation with sample images"""
    
    # Path to sample images
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(backend_dir, "uploads")
    
    # Find sample images
    sample_images = []
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(uploads_dir, file))
    
    if len(sample_images) < 3:
        print("❌ Not enough images for testing (need at least 3 images)")
        print("📁 Place at least 3 images in uploads folder")
        return
    
    # Use first 3 images for testing
    el1_path = sample_images[0]
    el2_path = sample_images[1] if len(sample_images) > 1 else sample_images[0]
    elbg_path = sample_images[2] if len(sample_images) > 2 else sample_images[0]
    
    print("🧪 Starting SNR and Sharpness calculation test...")
    print(f"📸 Images used:")
    print(f"   - EL1: {os.path.basename(el1_path)}")
    print(f"   - EL2: {os.path.basename(el2_path)}")
    print(f"   - ELBG: {os.path.basename(elbg_path)}")
    print()
    
    try:
        # Run quality check
        result = run_quality_check(el1_path, el2_path, elbg_path)
        
        print("\n🎯 Results summary:")
        print(f"   - SNR: {result['snr50']:.2f}")
        print(f"   - Sharpness (mm): {result['sharpness_mm']:.1f}")
        print(f"   - Category: {result['sharpness_category']}")
        print(f"   - Accepted: {'Yes' if result['accepted'] else 'No'}")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_with_sample_images()