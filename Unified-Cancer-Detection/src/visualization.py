import cv2
import numpy as np
from utils.config import Config

def overlay_contours(image, contours_info, default_is_malignant=False):
    """
    Overlay contours on the image.
    Color-coded: green for benign tumors, red for malignant tumors.
    """
    output = image.copy()
    
    for info in contours_info:
        cnt = info['contour']
        is_mal = info.get('is_malignant', default_is_malignant)
        
        # Color Code: Green for Benign, Red for Malignant
        color = Config.COLOR_MALIGNANT if is_mal else Config.COLOR_BENIGN
        
        # Clinical standard thickness
        cv2.drawContours(output, [cnt], -1, color, 2) 
        
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
