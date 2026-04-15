import cv2
import numpy as np
from utils.config import Config

def overlay_contours(image, contours_info, default_is_malignant=False):
    """
    Overlay contours on the image with thick borders and bounding boxes.
    Color-coded: green for benign tumors, red for malignant tumors.
    """
    output = image.copy()
    
    for info in contours_info:
        cnt = info['contour']
        is_mal = info.get('is_malignant', default_is_malignant)
        
        # Color Code: Green for Benign, Red for Malignant
        color = Config.COLOR_MALIGNANT if is_mal else Config.COLOR_BENIGN
        
        # Draw thick contour lines (5px for visibility)
        cv2.drawContours(output, [cnt], -1, color, 5)
        
        # Draw bounding box for additional clarity
        x, y, w, h = info['bbox']
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
        
        # Add a circle at the centroid
        cx, cy = info['centroid']
        cv2.circle(output, (cx, cy), 8, color, -1)
        cv2.circle(output, (cx, cy), 8, (255, 255, 255), 2)  # White outline
        
        # Add label text
        label = "MALIGNANT" if is_mal else "BENIGN"
        label_color = (0, 0, 255) if is_mal else (0, 255, 0)  # Red for mal, green for benign
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        
    return output

def save_output_image(image, save_path):
    """
    Saves the visualization to disk.
    """
    from PIL import Image
    if isinstance(image, np.ndarray):
        # image is assumed to be RGB
        img_pil = Image.fromarray(image)
    else:
        img_pil = image
    img_pil.save(save_path)
