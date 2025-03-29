import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def preprocess_image(img, size=(110, 110)):
    """
    Significantly enhanced preprocessing for much better hand gesture recognition
    
    Args:
        img: Input image (BGR format from OpenCV)
        size: Target size for resizing (default: 110x110 to maintain feature consistency)
    
    Returns:
        Preprocessed image and original processed image for display
    """
    try:
        # Step 1: Get the hand region from the image
        hand_region = get_hand_region(img)
        
        # Keep a copy for display purposes
        display_img = hand_region.copy()
        
        # Step 2: Multi-color space processing for robust skin detection
        
        # YCrCb color space (excellent for skin detection)
        ycrcb = cv2.cvtColor(hand_region, cv2.COLOR_BGR2YCrCb)
        
        # HSV color space (good for varying lighting conditions)
        hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
        
        # Lab color space (separates luminance from color, helpful for skin tone variations)
        lab = cv2.cvtColor(hand_region, cv2.COLOR_BGR2LAB)
        
        # Create multiple skin masks using different color spaces for robustness
        
        # YCrCb mask (wider range for better detection)
        lower_skin_ycrcb = np.array([0, 130, 75], dtype=np.uint8) 
        upper_skin_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # HSV mask (adjusted for better skin detection)
        lower_skin_hsv = np.array([0, 20, 40], dtype=np.uint8)
        upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)
        
        # Secondary HSV mask to capture broader skin tones
        lower_skin_hsv2 = np.array([170, 20, 40], dtype=np.uint8)
        upper_skin_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_skin_hsv2, upper_skin_hsv2)
        
        # Combine all masks for comprehensive skin detection
        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
        skin_mask = cv2.bitwise_or(skin_mask, mask_hsv2)
        
        # Apply enhanced morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Apply Gaussian blur to reduce noise and improve edge detection
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # Apply the skin mask to the original image
        skin = cv2.bitwise_and(hand_region, hand_region, mask=skin_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for better feature visibility
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Apply adaptive threshold for better edge detection in variable lighting
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 3)
        
        # Use Canny edge detection to enhance boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine edges with thresholded image for better feature representation
        combined = cv2.bitwise_or(thresh, edges)
        
        # Fill in the combined image to make solid regions
        kernel = np.ones((5,5), np.uint8)
        closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Resize to the target size
        resized = cv2.resize(closing, size)
        
        return resized, display_img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        # Fallback to basic preprocessing if the advanced method fails
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized = cv2.resize(threshold, size)
        return resized, img

def extract_features(img):
    """
    Significantly enhanced feature extraction with multiple advanced techniques
    tailored for rock, paper, scissors gesture recognition with improved accuracy
    
    Args:
        img: Preprocessed image
    
    Returns:
        Feature vector combining multiple feature types
    """
    features = []
    
    try:
        # 1. Basic pixel features (normalized)
        # Downsample to reduce dimensionality while keeping pattern
        h, w = img.shape
        downsampled = cv2.resize(img, (w//4, h//4))
        pixel_features = downsampled.flatten() / 255.0
        features.append(pixel_features)
        
        # 2. Histogram of Oriented Gradients (HOG) features - Fixed to ensure compatibility
        h, w = img.shape
        
        # Make sure HOG parameters are valid for the image size
        # Required: (winSize.width - blockSize.width) % blockStride.width == 0
        cell_size = 8  # cell size
        block_size = 2  # cells per block
        
        # Ensure proper alignment for HOG algorithm
        if h >= 32 and w >= 32:
            # Calculate valid dimensions that will work with these parameters
            # Making sure winSize - blockSize is divisible by blockStride
            block_pixels = block_size * cell_size
            stride_pixels = cell_size
            
            # Adjust width and height to be compatible with HOG parameters
            w_adjusted = ((w // stride_pixels) * stride_pixels)
            h_adjusted = ((h // stride_pixels) * stride_pixels)
            
            if w_adjusted >= block_pixels and h_adjusted >= block_pixels:
                # Resize image to adjusted dimensions if needed
                if w != w_adjusted or h != h_adjusted:
                    img_resized = cv2.resize(img, (w_adjusted, h_adjusted))
                else:
                    img_resized = img
                
                try:
                    # Configure HOG with values known to work
                    hog = cv2.HOGDescriptor(
                        (w_adjusted, h_adjusted),                    # winSize
                        (block_pixels, block_pixels),                # blockSize
                        (stride_pixels, stride_pixels),              # blockStride
                        (cell_size, cell_size),                      # cellSize
                        9                                           # nbins
                    )
                    
                    # Compute HOG features
                    hog_features = hog.compute(img_resized)
                    if hog_features is not None and hog_features.size > 0:
                        hog_features = hog_features.flatten()
                        features.append(hog_features)
                except Exception as hog_error:
                    logger.warning(f"HOG feature extraction error: {str(hog_error)}")
                    # If HOG fails, add basic gradient features instead
                    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
                    mag, ang = cv2.cartToPolar(gx, gy)
                    # Get histogram of gradient magnitudes - simpler than HOG but still useful
                    hist_mag = cv2.calcHist([mag], [0], None, [16], [0, np.pi*2])
                    features.append(hist_mag.flatten())
        
        # 3. Contour and shape features - critical for hand gesture recognition
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_features = []
        
        if contours:
            # Get the largest contour (should be the hand)
            max_contour = max(contours, key=cv2.contourArea)
            
            # Basic contour metrics
            area = cv2.contourArea(max_contour)
            perimeter = cv2.arcLength(max_contour, True)
            
            # Convex hull features
            hull = cv2.convexHull(max_contour)
            hull_area = cv2.contourArea(hull)
            
            # Critical for differentiating rock/paper/scissors: convexity defects
            # These are the "gaps" in the contour which help identify fingers
            hull_indices = cv2.convexHull(max_contour, returnPoints=False)
            if len(hull_indices) > 3 and len(max_contour) > 3:
                try:
                    defects = cv2.convexityDefects(max_contour, hull_indices)
                    if defects is not None:
                        # Count significant defects (potential finger gaps)
                        significant_defects = 0
                        defect_depths = []
                        
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            depth = d / 256.0  # Normalize depth
                            if depth > 5:  # Lower threshold to detect more finger gaps
                                significant_defects += 1
                                defect_depths.append(depth)
                        
                        # Average depth of defects
                        avg_depth = np.mean(defect_depths) if defect_depths else 0
                        max_depth = np.max(defect_depths) if defect_depths else 0
                        
                        # Add these to contour features - very useful for distinguishing gestures
                        contour_features.extend([significant_defects, avg_depth, max_depth])
                except:
                    # If convexity defects calculation fails, add zeros
                    contour_features.extend([0, 0, 0])
            else:
                contour_features.extend([0, 0, 0])
            
            # Calculate solidity (area / hull_area) - useful for rock vs paper
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # Aspect ratio of bounding rectangle - helps differentiate scissors
            x, y, w, h = cv2.boundingRect(max_contour)
            aspect_ratio = float(w) / h if h > 0 else 1
            
            # Extent: ratio of contour area to bounding rectangle area
            extent = float(area) / (w * h) if w * h > 0 else 0
            
            # Calculate circularity: 4*pi*area/perimeter^2
            # Close to 1 for circular objects (like rock gesture)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate equivalent diameter
            equi_diameter = np.sqrt(4 * area / np.pi)
            
            # Append basic shape features
            contour_features.extend([
                area / (img.shape[0] * img.shape[1]),  # Normalize area by image size
                perimeter / (img.shape[0] + img.shape[1]),  # Normalize perimeter
                solidity,
                aspect_ratio,
                extent,
                circularity,
                equi_diameter / max(img.shape[0], img.shape[1])  # Normalize diameter
            ])
            
            # Calculate moments and Hu moments - invariant to translation, rotation, scale
            moments = cv2.moments(max_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Take log to reduce range and improve model training
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            contour_features.extend(hu_moments)
            
            # Simplified contour (fewer points) to reduce noise
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(max_contour, epsilon, True)
            
            # Number of vertices in simplified contour
            # Helps distinguish rock (fewer corners) from scissors/paper
            contour_features.append(len(approx) / 100.0)  # Normalize
            
            # Add contour features
            features.append(np.array(contour_features))
        
        # 4. Region features: divide image into regions and compute statistics
        regions_h = 3
        regions_w = 3
        region_features = []
        
        # Divide image into regions and compute mean and std for each
        for i in range(regions_h):
            for j in range(regions_w):
                region = img[i*h//regions_h:(i+1)*h//regions_h, 
                             j*w//regions_w:(j+1)*w//regions_w]
                if region.size > 0:
                    region_mean = np.mean(region) / 255.0
                    region_std = np.std(region) / 255.0
                    region_features.extend([region_mean, region_std])
        
        if region_features:
            features.append(np.array(region_features))
        
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
    Significantly improved hand detection with multi-stage approach
    
    Args:
        img: Input image (BGR format from OpenCV)
        
    Returns:
        Tuple of (detection_success, processed_image)
    """
    try:
        # First approach: Multi-color space skin detection
        
        # Convert to multiple color spaces
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # YCrCb mask (wider range)
        lower_skin_ycrcb = np.array([0, 130, 75], dtype=np.uint8)
        upper_skin_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
        
        # HSV masks (multiple ranges to capture different skin tones)
        lower_skin_hsv1 = np.array([0, 20, 40], dtype=np.uint8)
        upper_skin_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower_skin_hsv1, upper_skin_hsv1)
        
        lower_skin_hsv2 = np.array([170, 20, 40], dtype=np.uint8)
        upper_skin_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_skin_hsv2, upper_skin_hsv2)
        
        # Combine masks
        skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv1)
        skin_mask = cv2.bitwise_or(skin_mask, mask_hsv2)
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # Find contours
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (assuming it's the hand)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # If area is reasonable, it's probably a hand
            # Lowered threshold to detect smaller hands or hands further from camera
            if area > 2000:
                # Create mask for largest contour
                mask = np.zeros_like(skin_mask)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                
                # Apply to original image
                result = cv2.bitwise_and(img, img, mask=mask)
                
                return True, result
        
        # Second approach: Background subtraction and motion detection
        # This assumes the hand is moving and is the main moving object
        
        # Convert to grayscale and apply blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Apply threshold to find high contrast regions (potential hand)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # If it's large enough, it might be a hand
            if cv2.contourArea(largest_contour) > 2000:
                mask = np.zeros_like(thresh)
                cv2.drawContours(mask, [largest_contour], 0, 255, -1)
                result = cv2.bitwise_and(img, img, mask=mask)
                return True, result
        
        # If all detection methods fail, return original image
        return False, img
    
    except Exception as e:
        logger.error(f"Error in detect_hand: {str(e)}")
        return False, img

def get_hand_region(img):
    """
    Extract and normalize the region of interest containing the hand
    with enhanced cropping and pre-processing
    
    Args:
        img: Input image (BGR format from OpenCV)
    
    Returns:
        Image cropped to the hand region with enhanced processing
    """
    try:
        # First attempt with multi-space skin-color based detection
        detected, hand_img = detect_hand(img)
        
        if not detected:
            # Try adaptive thresholding as a fallback
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 13, 4)
            
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
            # Convert to grayscale for bounding box detection
            gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            
            # Find non-zero pixels (the hand region)
            non_zero = cv2.findNonZero(gray)
            
            if non_zero is not None and len(non_zero) > 0:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(non_zero)
                
                # Add padding for better recognition (more context)
                padding = 40  # Increased padding
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)
                
                # Make the crop square by taking the larger dimension
                # Square crops help maintain aspect ratio for consistent recognition
                size = max(w, h)
                
                # Ensure within image boundaries
                x_center = x + w // 2
                y_center = y + h // 2
                half_size = size // 2
                
                x = max(0, x_center - half_size)
                y = max(0, y_center - half_size)
                size = min(size, 2 * (img.shape[1] - x), 2 * (img.shape[0] - y))
                
                # Crop the image to the hand region as a square
                cropped = img[y:y+size, x:x+size]
                
                if cropped.size == 0:
                    # If cropping failed, use original image
                    return img
                
                # Apply advanced enhancement techniques for better feature extraction
                
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel (enhanced contrast)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back to BGR
                enhanced_lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # Apply slight sharpening for better edge definition
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                
                # Apply slight noise reduction
                denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 5, 5, 7, 21)
                
                return denoised
                
        # If all else fails, return the original image
        return img
        
    except Exception as e:
        logger.error(f"Error in get_hand_region: {str(e)}")
        return img
