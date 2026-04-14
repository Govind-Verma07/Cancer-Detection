import cv2
import os
import numpy as np

def split_into_patches(image, patch_size=512, stride=512):
    """
    Splits a high-resolution image into patches.
    Returns a list of patches and their relative coordinates.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    h, w = image.shape[:2]
    
    # Pad if smaller than patch_size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        h, w = image.shape[:2]

    patches = []
    metadata = [] # (x_offset, y_offset, patch_id)
    
    patch_id = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            metadata.append({
                'patch_id': patch_id,
                'x_offset': x,
                'y_offset': y,
                'width': patch_size,
                'height': patch_size
            })
            patch_id += 1
            
    # If still empty (shouldn't happen with padding), add one patch
    if not patches:
        patches.append(image[:patch_size, :patch_size])
        metadata.append({'patch_id': 0, 'x_offset': 0, 'y_offset': 0, 'width': patch_size, 'height': patch_size})

    return patches, metadata

def reconstruct_from_patches(patches, metadata, original_shape):
    """
    Stitches patches back into a full-sized image (or mask).
    """
    if not patches:
        return np.zeros(original_shape, dtype=np.uint8)

    h, w = original_shape[:2]
    full_image = np.zeros(original_shape, dtype=patches[0].dtype)
    
    for patch, meta in zip(patches, metadata):
        x, y = meta['x_offset'], meta['y_offset']
        ph, pw = patch.shape[:2]
        # Crop patch if it extends beyond the original shape (due to padding)
        h_fit = min(ph, h - y)
        w_fit = min(pw, w - x)
        
        if h_fit > 0 and w_fit > 0:
            full_image[y:y+h_fit, x:x+w_fit] = patch[:h_fit, :w_fit]
        
    return full_image
