import cv2
import numpy as np
import time
import os
from datetime import datetime


def validate_frame(frame, frame_number):
    """
    Validate a video frame and return diagnostic information.
    
    Args:
        frame: The OpenCV frame to validate
        frame_number: The frame number
        
    Returns:
        dict: Dictionary with frame validation results
    """
    if frame is None:
        return {
            'valid': False,
            'error': 'Frame is None',
            'frame_number': frame_number
        }
    
    # Check frame dimensions
    height, width = frame.shape[:2]
    
    # Check if frame has content (not completely black or white)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    brightness = np.mean(gray)
    std_dev = np.std(gray)
    has_content = std_dev > 5.0  # Threshold for determining if frame has content
    
    # Check for corruption (NaN values)
    has_nan = np.isnan(frame).any()
    
    return {
        'valid': not has_nan and has_content,
        'frame_number': frame_number,
        'dimensions': (width, height),
        'brightness': brightness,
        'std_dev': std_dev,
        'has_content': has_content,
        'has_nan': has_nan
    }


def save_frame_sample(frame, frame_number, output_dir="frame_samples"):
    """
    Save a sample frame to disk for manual inspection.
    
    Args:
        frame: OpenCV frame to save
        frame_number: Frame number for filename
        output_dir: Directory where frames should be saved
    
    Returns:
        str: Path to saved frame or None if saving failed
    """
    if frame is None:
        return None
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/frame_{frame_number}_{timestamp}.jpg"
    
    try:
        cv2.imwrite(filename, frame)
        return filename
    except Exception as e:
        print(f"Error saving frame: {str(e)}")
        return None


def draw_diagnostics(frame, diagnostics):
    """
    Draw diagnostic information on the frame.
    
    Args:
        frame: The OpenCV frame to annotate
        diagnostics: Dictionary with diagnostic information
        
    Returns:
        frame: The annotated frame
    """
    if frame is None:
        return None
        
    # Create a copy of the frame
    annotated = frame.copy()
    
    # Draw frame number
    cv2.putText(
        annotated,
        f"Frame: {diagnostics['frame_number']}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )
    
    # Draw dimensions
    cv2.putText(
        annotated,
        f"Dimensions: {diagnostics['dimensions'][0]}x{diagnostics['dimensions'][1]}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1
    )
    
    # Draw brightness info
    cv2.putText(
        annotated,
        f"Brightness: {diagnostics['brightness']:.1f}, StdDev: {diagnostics['std_dev']:.1f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        1
    )
    
    # Draw validation status
    status_color = (0, 255, 0) if diagnostics['valid'] else (0, 0, 255)
    cv2.putText(
        annotated,
        f"Valid: {diagnostics['valid']}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        1
    )
    
    # Draw timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        annotated,
        timestamp,
        (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )
    
    return annotated 