import numpy as np
from PIL import Image
import cv2
from scipy.stats import skew, kurtosis

# =========================
# Utility Functions
# =========================
def load_image_as_array(path):
    """Load grayscale image as float64 array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float64)

def calculate_snr(el1, el2, elbg):
    """Calculate SNR50 according to IEC TS 60904-13:2018."""
    # Get the minimum dimensions to resize all images to the same size
    min_height = min(el1.shape[0], el2.shape[0], elbg.shape[0])
    min_width = min(el1.shape[1], el2.shape[1], elbg.shape[1])
    
    # Resize all images to the same size
    el1_resized = cv2.resize(el1, (min_width, min_height))
    el2_resized = cv2.resize(el2, (min_width, min_height))
    elbg_resized = cv2.resize(elbg, (min_width, min_height))
    
    signal = 0.5 * (el1_resized + el2_resized) - elbg_resized
    noise = np.abs(el1_resized - el2_resized) * (np.sqrt(2 / np.pi) ** -0.5)
    snr50 = np.sum(signal) / np.sum(noise)
    return snr50, signal

def sharpen_image(image):
    """Sharpen the image using a kernel filter."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image.astype(np.uint8), -1, kernel)
    return sharpened

def histogram_analysis(image):
    """Compute mean, variance, skewness, and kurtosis of histogram."""
    flat = image.flatten()
    mean_val = np.mean(flat)
    var_val = np.var(flat)
    skew_val = skew(flat)
    kurt_val = kurtosis(flat)
    return mean_val, var_val, skew_val, kurt_val

# =========================
# Main Pipeline
# =========================
def el_pipeline(el1_path, el2_path, elbg_path):
    # Load images
    EL1 = load_image_as_array(el1_path)
    EL2 = load_image_as_array(el2_path)
    ELBG = load_image_as_array(elbg_path)

    # Calculate SNR50 and get final EL image
    snr50, el_final = calculate_snr(EL1, EL2, ELBG)

    # Normalize to 8-bit range for saving/viewing
    el_final_norm = cv2.normalize(el_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply sharpening
    el_sharpened = sharpen_image(el_final_norm)

    # Histogram analysis
    mean_val, var_val, skew_val, kurt_val = histogram_analysis(el_final_norm)

    # Save results
    cv2.imwrite("EL_final.png", el_final_norm)
    cv2.imwrite("EL_sharpened.png", el_sharpened)

    print(f"[INFO] SNR50 = {snr50:.2f}")
    print(f"[INFO] Histogram Analysis -> Mean: {mean_val:.2f}, Var: {var_val:.2f}, Skew: {skew_val:.2f}, Kurtosis: {kurt_val:.2f}")
    print("[INFO] Saved 'EL_final.png' and 'EL_sharpened.png'.")

    return snr50, (mean_val, var_val, skew_val, kurt_val), el_final_norm, el_sharpened


# =========================
# Example Usage
# =========================
el1_path = r'C:\Users\DELL\Downloads\crop.v1-roboflow-instant-1--eval-.yolov8\ISC_1.JPG'
el2_path = r'C:\Users\DELL\Downloads\crop.v1-roboflow-instant-1--eval-.yolov8\ISC_2.JPG'
elbg_path = r'C:\Users\DELL\Downloads\crop.v1-roboflow-instant-1--eval-.yolov8\B_G.JPG'

snr_value, hist_stats, el_final, el_sharpened = el_pipeline(el1_path, el2_path, elbg_path)
