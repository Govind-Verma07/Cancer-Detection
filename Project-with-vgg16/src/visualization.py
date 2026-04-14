import cv2
import numpy as np
from utils.config import Config

def overlay_contours(image, contours_info, default_is_malignant=False):
    """
    Overlay contours on the image.
    Individual color coding per contour based on tumor Stage.
    """
    output = image.copy()
    
    for info in contours_info:
        cnt = info['contour']
        is_mal = info.get('is_malignant', default_is_malignant)
        
        # Color Code based on Stage
        if not is_mal:
            color = Config.COLOR_BENIGN
        else:
            stage_str = str(info.get('stage', 'Stage 3'))
            if "1" in stage_str:
                color = Config.COLOR_STAGE_1
            elif "2" in stage_str:
                color = Config.COLOR_STAGE_2
            elif "3" in stage_str:
                color = Config.COLOR_STAGE_3
            elif "4" in stage_str:
                color = Config.COLOR_STAGE_4
            else:
                color = Config.COLOR_STAGE_3 # fallback
        
        # Clinical standard thickness (decreased border size for precision)
        cv2.drawContours(output, [cnt], -1, color, 1) 
        
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
