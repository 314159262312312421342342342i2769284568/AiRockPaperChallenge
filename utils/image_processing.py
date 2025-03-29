import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_image(img, size=(128, 128)):
    """
    Enhanced preprocessing for better hand gesture recognition
    
    Args:
        img: Input image (BGR format from OpenCV)
        size: Target size for resizing (default: 128x128 for more detail)
    
    Returns:
        Preprocessed image
    """
    try:
        # Step 1: Get the hand region from the image
        hand_region = get_hand_region(img)
        
        # Step 2: Convert to different color spaces for feature extraction
        # YCrCb color space can help with skin detection
        ycrcb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
        
        # HSV color space is also useful for skin segmentation
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Extract skin mask using YCrCb color space (typical skin values)
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Combine with HSV mask for better skin segmentation
        lower_skin_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin_hsv = np.array([20, 150, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to reduce noise
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # Apply the skin mask to the original image
        skin = cv2.bitwise_and(hand_region, hand_region, mask=skin_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold for better edge detection in variable lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Resize to the target size
        resized = cv2.resize(thresh, size)
        
        return resized
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        # Fallback to basic preprocessing if the advanced method fails
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized = cv2.resize(threshold, size)
        return resized

def extract_features(img):
    """
    Enhanced feature extraction with multiple techniques
    
    Args:
        img: Preprocessed image
    
    Returns:
        Feature vector combining multiple feature types
    """
    features = []
    
    try:
        # 1. Basic pixel features (normalized)
        pixel_features = img.flatten() / 255.0
        features.append(pixel_features)
        
        # 2. Histogram of Oriented Gradients (HOG) features
        # Calculate HOG features - captures shape information
        # Use smaller cells for finer detail on hand shapes
        h, w = img.shape
        cell_size = 8  # smaller cells capture more detail
        block_size = 2  # number of cells per block
        
        if h >= 32 and w >= 32:  # Ensure image is large enough for HOG
            hog = cv2.HOGDescriptor(
                (w, h),                   # winSize
                (block_size * cell_size, block_size * cell_size),  # blockSize
                (cell_size, cell_size),  # blockStride
                (cell_size, cell_size),  # cellSize
                9                        # nbins
            )
            hog_features = hog.compute(img)
            if hog_features is not None:
                hog_features = hog_features.flatten()
                features.append(hog_features)
        
        # 3. Add contour features for shape analysis
        # Find contours
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_features = []
        
        if contours:
            # Get the largest contour (should be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Area and perimeter
            area = cv2.contourArea(max_contour)
            perimeter = cv2.arcLength(max_contour, True)
            
            # Convex hull features
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            
            # Calculate solidity (area / hull_area)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Calculate convexity (hull_perimeter / perimeter)
            hull_perimeter = cv2.arcLength(hull, True)
            convexity = hull_perimeter / perimeter if perimeter > 0 else 0
            
            # Calculate moments and center of mass
            moments = cv2.moments(max_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = 0, 0
                
            # Append shape features
            contour_features = [
                area, 
                perimeter, 
                solidity, 
                convexity,
                hull_area,
                cx / w if w > 0 else 0,  # Normalize by image width
                cy / h if h > 0 else 0   # Normalize by image height
            ]
            
            features.append(np.array(contour_features))
        
        # Combine all features
        combined_features = np.concatenate([f for f in features if len(f) > 0])
        
        # Normalize combined features
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
            
        return combined_features
        
    except Exception as e:
        logger.error(f"Error in extract_features: {str(e)}")
        # Fallback to basic features
        return img.flatten() / 255.0

def detect_hand(img):
    """
    Improved hand detection with skin color segmentation
    
    Args:
        img: Input image (BGR format from OpenCV)
        
    Returns:
        Tuple of (detection_success, processed_image)
    """
    try:
        # Convert to YCrCb color space (good for skin detection)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 135, 85], dtype=np.uint8)
        upper_skin = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return False
        if not contours:
            return False, img
        
        # Find the largest contour (assuming it's the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # If the contour area is too small, it's probably not a hand
        if cv2.contourArea(largest_contour) < 3000:  # Lower threshold to detect smaller hands
            return False, img
        
        # Create a mask for the hand
        mask = np.zeros_like(skin_mask)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        result = cv2.bitwise_and(img, img, mask=mask)
        
        return True, result
        
    except Exception as e:
        logger.error(f"Error in detect_hand: {str(e)}")
        return False, img

def get_hand_region(img):
    """
    Extract and normalize the region of interest containing the hand
    
    Args:
        img: Input image (BGR format from OpenCV)
    
    Returns:
        Image cropped to the hand region
    """
    try:
        # First attempt with skin-color based detection
        detected, hand_img = detect_hand(img)
        
        if not detected:
            # Try alternative approach with adaptive thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create a mask
                mask = np.zeros_like(thresh)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                
                # Apply the mask to the original image
                hand_img = cv2.bitwise_and(img, img, mask=mask)
                detected = True
            else:
                # If still not detected, use the original image
                return img
        
        if detected:
            # Convert to grayscale
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            
            # Find non-zero pixels (the hand)
            non_zero = cv2.findNonZero(gray)
            
            if non_zero is not None:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(non_zero)
                
                # Add padding for better recognition
                padding = 30  # Increased padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)
                
                # Make the crop square by taking the larger dimension
                size = max(w, h)
                # Ensure within image boundaries
                size = min(size, img.shape[0] - y, img.shape[1] - x)
                
                # Crop the image to the hand region as a square
                cropped = img[y:y+size, x:x+size]
                
                # Apply contrast enhancement
                lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                return enhanced
        
        return img  # Return original if all else fails
        
    except Exception as e:
        logger.error(f"Error in get_hand_region: {str(e)}")
        return img
