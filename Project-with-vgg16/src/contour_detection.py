import cv2
import numpy as np

def detect_contours(mask):
    """
    Detect contours from a binary mask.
    Returns a list of contours and their properties.
    """
    # Ensure mask is uint8 and binary (0 or 255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contour_stats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Size Filter (Decreased sensitivity to filter out small artifacts/skin tissue)
        if area < 1000: # Increased to 1000 to filter out artifacts
            continue
            
        # 2. Shape Filter (Circularity & Solidity)
        # Circularity = 4 * pi * Area / Perimeter^2 (1.0 is perfect circle)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        # Solidity = Area / Convex Hull Area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Lumps are compact (high circularity/solidity), nerves are elongated (low)
        # BUGFIX: Malignant tumors are highly irregular (spiculate), so circularity/solidity can be extremely low!
        # Drastically relaxed geometric restrictions.
        if circularity < 0.05 or solidity < 0.2:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
            
        contour_stats.append({
            'contour': cnt,
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'area': area
        })
        
    return contour_stats
