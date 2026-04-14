import cv2
import numpy as np

def detect_contours(mask, architecture="resnet50"):
    """
    Detect contours from a binary mask with architecture-specific filtering.
    """
    # Ensure mask is uint8 and binary (0 or 255)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    
    # Selection of algorithm based on architecture
    mode = cv2.CHAIN_APPROX_SIMPLE if architecture == "resnet50" else cv2.CHAIN_APPROX_NONE
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, mode)
    
    # Parameter configuration
    if architecture == "resnet50":
        min_area = 1000
        min_circularity = 0.15   # Relaxed to resolve "Blank" prediction issues
        min_solidity = 0.20
    else: # vgg16
        min_area = 1000
        min_circularity = 0.70   # Strictly circular (Filters out elongated nerve cells)
        min_solidity = 0.60      # Dense lumps only
        
    contour_stats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # 1. Size Filter
        if area < min_area: 
            continue
            
        # 2. Shape Filter (Circularity & Solidity)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Apply specialized filters
        if circularity < min_circularity or solidity < min_solidity:
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
