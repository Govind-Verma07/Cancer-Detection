import numpy as np
import cv2

def calculate_tumor_metrics(mask, image_rgb, threshold=0.5):
    """
    Calculates breast-to-tumor area ratio based on estimated breast region.
    """
    import cv2
    grayscale = np.mean(image_rgb, axis=2).astype(np.uint8)
    # Otsu's thresholding to find the breast tissue automatically
    _, breast_mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    breast_area = np.sum(breast_mask > 0)
    tumor_area = np.sum(mask > threshold)
    
    ratio = tumor_area / breast_area if breast_area > 0 else 0
    
    return {
        'tumor_area': int(tumor_area),
        'breast_area': int(breast_area),
        'ratio': ratio
    }
    
    return {
        'tumor_area': int(tumor_area),
        'total_area': total_area,
        'ratio': ratio
    }

def get_relative_coordinates(centroid, image_shape):
    """
    Converts absolute centroid (x, y) to relative (0-1).
    """
    cx, cy = centroid
    h, w = image_shape[:2]
    return cx / w, cy / h

def get_formatted_coordinates(rel_coord):
    """
    Returns relative coordinates as percentage strings.
    """
    rx, ry = rel_coord
    return f"X: {rx*100:.1f}%, Y: {ry*100:.1f}%"
