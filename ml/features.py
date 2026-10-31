import os
import io
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from scipy.ndimage import gaussian_filter

def preprocess_image(img_input, target_size=(256, 256)):
    """
    Standardize drawing images to uniform dimensions, grayscale, and normalized contrast.
    Supports file path, numpy array, bytes, or PIL image.
    Handles RGB, RGBA, Grayscale, 1-channel, and diverse image formats.
    """
    if isinstance(img_input, str):
        image = cv2.imread(img_input, cv2.IMREAD_UNCHANGED)
        if image is None:
            # Fallback to PIL
            image = np.array(Image.open(img_input))
    elif isinstance(img_input, bytes):
        data = np.frombuffer(img_input, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.array(Image.open(io.BytesIO(img_input)))
    elif isinstance(img_input, Image.Image):
        image = np.array(img_input)
    elif isinstance(img_input, np.ndarray):
        image = img_input.copy()
    else:
        image = np.array(img_input)

    if image is None or (isinstance(image, np.ndarray) and image.size == 0):
        raise ValueError("Empty or invalid image data")

    # If single channel with shape (H, W, 1), squeeze to (H, W)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze(axis=2)

    # Handle color channels
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            alpha = image[:, :, 3].astype(float) / 255.0
            rgb = image[:, :, :3].astype(float)
            white_bg = np.ones_like(rgb) * 255.0
            composite = (rgb * alpha[:, :, None] + white_bg * (1.0 - alpha[:, :, None])).astype(np.uint8)
            gray = cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 3:  # RGB / BGR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image[:, :, 0].copy()
    elif len(image.shape) == 2:
        gray = image.copy()
    else:
        raise ValueError(f"Unsupported image dimensions: {image.shape}")

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    # Resize preserving drawing geometry
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    # Invert if the background is dark and strokes are light (e.g. tablet / inverted canvas)
    mean_val = np.mean(resized)
    if mean_val < 127:
        resized = 255 - resized

    # Normalize contrast
    norm_img = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm_img

def extract_hog_features(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    """
    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        transform_sqrt=True,
        block_norm="L2-Hys"
    )
    return features

def extract_biomechanical_metrics(image):
    """
    Extract domain-specific tremor and kinematic irregularity biomarkers:
    - Tremor roughness & curvature variance
    - Line thickness variance
    - 2D FFT spectral high-frequency energy ratio
    - Gradient entropy (directional randomness)
    - Stroke-to-envelope ratio
    Safe against empty contours, blank images, and non-standard strokes.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Binary thresholding for stroke isolation (Otsu)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    try:
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    if binary is None or np.sum(binary) == 0:
        binary = np.where(blurred < 220, 255, 0).astype(np.uint8)

    # 1. Stroke count and contour analysis
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    total_length = 0.0
    curvature_variances = []
    jaggedness_scores = []

    for cnt in contours:
        if len(cnt) < 10:
            continue
        total_length += cv2.arcLength(cnt, closed=False)
        pts = cnt.squeeze()
        if len(pts.shape) != 2 or pts.shape[0] < 5:
            continue

        # Compute point-to-point displacement angles (direction changes)
        dx = np.diff(pts[:, 0].astype(float))
        dy = np.diff(pts[:, 1].astype(float))
        angles = np.arctan2(dy, dx)
        angle_diffs = np.abs(np.diff(angles))
        angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
        
        if len(angle_diffs) > 0:
            var_val = float(np.var(angle_diffs))
            if not np.isnan(var_val):
                curvature_variances.append(var_val)
            sharp_changes = np.sum(angle_diffs > (np.pi / 4)) / max(len(angle_diffs), 1)
            jaggedness_scores.append(float(sharp_changes))

    avg_curvature_var = float(np.mean(curvature_variances)) if curvature_variances else 0.0
    avg_jaggedness = float(np.mean(jaggedness_scores)) if jaggedness_scores else 0.0

    # 2. Spectral Analysis (FFT 2D) - Tremors generate high-frequency micro-deviations
    f = np.fft.fft2(image.astype(float))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    h, w = image.shape
    cy, cx = h // 2, w // 2
    r_inner = max(min(h, w) // 8, 1)
    r_outer = max(min(h, w) // 3, r_inner + 1)

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    low_freq_mask = dist_from_center <= r_inner
    high_freq_mask = (dist_from_center > r_inner) & (dist_from_center <= r_outer)
    
    low_freq_energy = float(np.sum(magnitude_spectrum[low_freq_mask])) + 1e-6
    high_freq_energy = float(np.sum(magnitude_spectrum[high_freq_mask]))
    spectral_energy_ratio = float(high_freq_energy / low_freq_energy)
    if np.isnan(spectral_energy_ratio):
        spectral_energy_ratio = 0.0

    # 3. Stroke Thickness Inconsistency (Distance Transform)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    stroke_widths = dist_transform[binary > 0]
    stroke_width_std = float(np.std(stroke_widths)) if len(stroke_widths) > 0 else 0.0
    stroke_width_mean = float(np.mean(stroke_widths)) if len(stroke_widths) > 0 else 0.0
    if np.isnan(stroke_width_std):
        stroke_width_std = 0.0
    if np.isnan(stroke_width_mean):
        stroke_width_mean = 0.0

    # 4. Gradient Directional Entropy
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(sobelx, sobely)
    grad_dir = np.arctan2(sobely, sobelx)
    
    p85 = np.percentile(grad_mag, 85) if grad_mag.size > 0 else 0
    strong_edges = grad_mag > p85
    if np.sum(strong_edges) > 50:
        hist, _ = np.histogram(grad_dir[strong_edges], bins=16, range=(-np.pi, np.pi), density=True)
        hist = hist[hist > 0]
        grad_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
        if np.isnan(grad_entropy):
            grad_entropy = 0.0
    else:
        grad_entropy = 0.0

    # 5. Composite Tremor Index (Normalized 0 - 100)
    raw_tremor = (avg_curvature_var * 40.0) + (avg_jaggedness * 60.0) + (spectral_energy_ratio * 25.0) + (stroke_width_std * 5.0)
    if np.isnan(raw_tremor):
        raw_tremor = 0.0
    tremor_index = float(np.clip(raw_tremor * 100.0, 5.0, 98.0))

    return {
        "curvature_variance": round(avg_curvature_var, 5),
        "jaggedness_ratio": round(avg_jaggedness, 4),
        "spectral_energy_ratio": round(spectral_energy_ratio, 5),
        "stroke_width_std": round(stroke_width_std, 4),
        "stroke_width_mean": round(stroke_width_mean, 4),
        "grad_entropy": round(grad_entropy, 4),
        "total_contour_length": round(total_length, 2),
        "tremor_index": round(tremor_index, 1)
    }

def extract_combined_features(image):
    """
    Combine HOG features and tabular biomechanical tremor features.
    """
    hog_feats = extract_hog_features(image)
    bio = extract_biomechanical_metrics(image)
    bio_vector = np.array([
        bio["curvature_variance"],
        bio["jaggedness_ratio"],
        bio["spectral_energy_ratio"],
        bio["stroke_width_std"],
        bio["stroke_width_mean"],
        bio["grad_entropy"],
        bio["tremor_index"]
    ], dtype=np.float32)
    
    return np.hstack([hog_feats, bio_vector]), bio

def generate_tremor_heatmap(image):
    """
    Generates a color-coded tremor intensity overlay highlighting micro-jitters
    and deviations in patient drawings. Safe against empty/sparse masks.
    """
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur = cv2.GaussianBlur(mag.astype(float), (9, 9), 0)
    blur_sq = cv2.GaussianBlur(mag.astype(float)**2, (9, 9), 0)
    local_var = np.maximum(0, blur_sq - blur**2)
    local_var_norm = cv2.normalize(local_var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heatmap = cv2.applyColorMap(local_var_norm, cv2.COLORMAP_JET)

    # Mask with binary stroke
    try:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except Exception:
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    stroke_mask = cv2.dilate(binary, kernel, iterations=2)

    orig_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = orig_bgr.copy()
    
    mask_bool = stroke_mask > 0
    if np.sum(mask_bool) > 0:
        overlay[mask_bool] = cv2.addWeighted(orig_bgr[mask_bool], 0.35, heatmap[mask_bool], 0.65, 0)
    else:
        overlay = cv2.addWeighted(orig_bgr, 0.7, heatmap, 0.3, 0)

    return overlay
