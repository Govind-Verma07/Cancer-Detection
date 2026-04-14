import cv2
import numpy as np
from utils.config import Config

def overlay_contours(image, contours_info, default_is_malignant=False):
    """
    Overlay contours on the image.
    Individual color coding per contour if 'is_malignant' is in info.
    """
    output = image.copy()
    
    for info in contours_info:
        cnt = info['contour']
        is_mal = info.get('is_malignant', default_is_malignant)
        color = Config.COLOR_MALIGNANT if is_mal else Config.COLOR_BENIGN
        
        # Clinical standard thickness
        cv2.drawContours(output, [cnt], -1, color, 2) 
        
        # --- ADD TEXT LABELS ---
        x, y, w, h = info.get('bbox', cv2.boundingRect(cnt))
        label = "MALIGNANT" if is_mal else "BENIGN"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Draw label background for readability
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(output, (x, y - label_h - 10), (x + label_w, y), color, -1)
        cv2.putText(output, label, (x, y - 5), font, font_scale, (255, 255, 255), thickness)
        
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
