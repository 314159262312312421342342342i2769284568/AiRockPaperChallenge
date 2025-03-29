import cv2
import numpy as np

def preprocess_image(img, size=(64, 64)):
    """
    Preprocess image for feature extraction
    
    Args:
        img: Input image (BGR format from OpenCV)
        size: Target size for resizing (default: 64x64)
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply threshold to create binary image
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Resize to a standard size
    resized = cv2.resize(threshold, size)
    
    return resized

def extract_features(img):
    """
    Extract features from processed image
    
    Args:
        img: Preprocessed image
    
    Returns:
        Feature vector
    """
    # For a simple approach, we'll just flatten the image and use pixel values as features
    features = img.flatten() / 255.0  # Normalize pixel values
    
    # You could add more sophisticated feature extraction here, such as:
    # - Histogram of Oriented Gradients (HOG)
    # - Local Binary Patterns (LBP)
    # - Edge detection features
    
    return features

def detect_hand(img):
    """
    Attempt to detect a hand in the image
    
    Args:
        img: Input image (BGR format from OpenCV)
        
    Returns:
        Tuple of (detection_success, processed_image)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to create binary image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return False
    if not contours:
        return False, img
    
    # Find the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # If the contour area is too small, it's probably not a hand
    if cv2.contourArea(largest_contour) < 5000:  # Adjust this threshold as needed
        return False, img
    
    # Create a mask for the hand
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    return True, result

def get_hand_region(img):
    """
    Extract the region of interest containing the hand
    
    Args:
        img: Input image (BGR format from OpenCV)
    
    Returns:
        Image cropped to the hand region
    """
    # Detect the hand
    detected, hand_img = detect_hand(img)
    
    if not detected:
        return img  # Return original if no hand detected
    
    # Convert to grayscale
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    
    # Find non-zero pixels (the hand)
    non_zero = cv2.findNonZero(gray)
    
    if non_zero is None:
        return img  # Return original if no non-zero pixels
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(non_zero)
    
    # Add padding for better recognition
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)
    
    # Crop the image to the hand region
    cropped = img[y:y+h, x:x+w]
    
    return cropped
