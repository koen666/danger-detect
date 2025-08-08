import cv2
import numpy as np


def preprocess_frame(frame, resize_dim=None, normalize=True, denoise=True, 
                     enhance=True, equalize_hist=False):
    """
    Preprocess a video frame with multiple operations:
    - Denoising (Gaussian or bilateral filter)
    - Enhancement (contrast and brightness adjustment)
    - Resizing
    - Normalization (pixel values to 0-1 range)
    
    Args:
        frame: Input OpenCV frame
        resize_dim: Tuple (width, height) to resize frame, None for no resize
        normalize: Whether to normalize pixel values to 0-1 range
        denoise: Whether to apply denoising filter
        enhance: Whether to enhance contrast and brightness
        equalize_hist: Whether to apply histogram equalization
        
    Returns:
        Preprocessed frame
    """
    if frame is None:
        return None
    
    # Create a copy to avoid modifying the original
    processed = frame.copy()
    
    # Apply denoising if requested
    if denoise:
        processed = apply_denoising(processed)
    
    # Apply enhancement if requested
    if enhance:
        processed = enhance_frame(processed)
    
    # Apply histogram equalization if requested
    if equalize_hist:
        processed = equalize_histogram(processed)
    
    # Resize if dimensions provided
    if resize_dim is not None:
        processed = resize_frame(processed, resize_dim)
    
    # Normalize pixel values if requested
    if normalize:
        processed = normalize_frame(processed)
    
    return processed


def apply_denoising(frame, method='gaussian', params=None):
    """
    Apply denoising to a frame.
    
    Args:
        frame: Input OpenCV frame
        method: Denoising method ('gaussian', 'bilateral', 'nlmeans')
        params: Parameters for the denoising method
        
    Returns:
        Denoised frame
    """
    if frame is None:
        return None
    
    # Default parameters
    if params is None:
        if method == 'gaussian':
            params = {'ksize': (5, 5), 'sigmaX': 0}
        elif method == 'bilateral':
            params = {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
        elif method == 'nlmeans':
            params = {'h': 10, 'templateWindowSize': 7, 'searchWindowSize': 21}
    
    # Apply selected denoising method
    if method == 'gaussian':
        return cv2.GaussianBlur(frame, params['ksize'], params['sigmaX'])
    elif method == 'bilateral':
        return cv2.bilateralFilter(frame, params['d'], params['sigmaColor'], params['sigmaSpace'])
    elif method == 'nlmeans':
        if len(frame.shape) == 3:  # Color image
            return cv2.fastNlMeansDenoisingColored(
                frame, None, params['h'], params['h'], 
                params['templateWindowSize'], params['searchWindowSize']
            )
        else:  # Grayscale
            return cv2.fastNlMeansDenoising(
                frame, None, params['h'], 
                params['templateWindowSize'], params['searchWindowSize']
            )
    else:
        print(f"Warning: Unknown denoising method '{method}'. Returning original frame.")
        return frame


def enhance_frame(frame, alpha=1.2, beta=10):
    """
    Enhance a frame by adjusting contrast and brightness.
    
    Args:
        frame: Input OpenCV frame
        alpha: Contrast control (1.0 means no change)
        beta: Brightness control (0 means no change)
        
    Returns:
        Enhanced frame
    """
    if frame is None:
        return None
    
    # Apply contrast and brightness adjustment
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return enhanced


def equalize_histogram(frame):
    """
    Apply histogram equalization to improve contrast.
    
    Args:
        frame: Input OpenCV frame
        
    Returns:
        Equalized frame
    """
    if frame is None:
        return None
    
    if len(frame.shape) == 3:  # Color image
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Apply equalization only to the Y channel
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        
        # Convert back to BGR
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return equalized
    else:  # Grayscale
        return cv2.equalizeHist(frame)


def resize_frame(frame, target_dim, interpolation=cv2.INTER_AREA):
    """
    Resize a frame to target dimensions.
    
    Args:
        frame: Input OpenCV frame
        target_dim: Tuple (width, height) of target dimensions
        interpolation: Interpolation method
        
    Returns:
        Resized frame
    """
    if frame is None:
        return None
    
    return cv2.resize(frame, target_dim, interpolation=interpolation)


def normalize_frame(frame, scale=1.0, dtype=np.float32):
    """
    Normalize pixel values to 0-1 range.
    
    Args:
        frame: Input OpenCV frame
        scale: Scale factor to apply after normalization
        dtype: Data type of output array
        
    Returns:
        Normalized frame
    """
    if frame is None:
        return None
    
    # Normalize to 0-1 range and scale
    normalized = (frame.astype(dtype) / 255.0) * scale
    return normalized


def adaptive_threshold(frame, block_size=11, c=2):
    """
    Apply adaptive thresholding to a grayscale frame.
    
    Args:
        frame: Grayscale input frame
        block_size: Size of pixel neighborhood for threshold calculation
        c: Constant subtracted from mean
        
    Returns:
        Binary threshold image
    """
    if frame is None:
        return None
    
    # Ensure input is grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c
    )
    
    return thresh


def edge_detection(frame, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection to a frame.
    
    Args:
        frame: Input OpenCV frame
        low_threshold: Lower threshold for edge detection
        high_threshold: Higher threshold for edge detection
        
    Returns:
        Edge map
    """
    if frame is None:
        return None
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def extract_hog_features(frame, win_size=(64, 64), block_size=(16, 16),
                         block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    
    Args:
        frame: Input frame (should be resized to appropriate dimensions)
        win_size: Detection window size
        block_size: Block size in pixels
        block_stride: Block stride in pixels
        cell_size: Cell size in pixels
        nbins: Number of bins for the histograms
        
    Returns:
        HOG descriptor for the frame
    """
    if frame is None:
        return None
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Initialize HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # Compute HOG features
    features = hog.compute(gray)
    
    return features


def rgb_to_hsv(frame):
    """
    Convert frame from RGB to HSV color space.
    
    Args:
        frame: Input BGR frame
        
    Returns:
        HSV frame
    """
    if frame is None or len(frame.shape) != 3:
        return None
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# Test function for the preprocessing module
if __name__ == "__main__":
    import os
    import time
    
    # Open a video source (camera or file)
    cap = cv2.VideoCapture(0)  # Use camera index 0
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        exit()
    
    # Create output directory for samples
    output_dir = "preprocessing_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing preprocessing module. Press 'q' to exit, 's' to save samples.")
    
    frame_count = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Create a list of different preprocessing combinations
        processed_frames = {
            "Original": frame.copy(),
            "Denoised": apply_denoising(frame),
            "Enhanced": enhance_frame(frame),
            "Equalized": equalize_histogram(frame),
            "Edge Detection": cv2.cvtColor(edge_detection(frame), cv2.COLOR_GRAY2BGR),
            "Full Preprocessing": preprocess_frame(frame, resize_dim=(640, 480))
        }
        
        # Create a grid for display
        rows = 2
        cols = 3
        cell_height = 240
        cell_width = 320
        grid = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
        
        # Place processed frames in the grid
        for i, (title, proc_frame) in enumerate(processed_frames.items()):
            r, c = i // cols, i % cols
            # Resize for display
            display_frame = cv2.resize(proc_frame, (cell_width, cell_height))
            
            # Add to grid
            grid[r*cell_height:(r+1)*cell_height, c*cell_width:(c+1)*cell_width] = display_frame
            
            # Add title
            cv2.putText(
                grid, 
                title, 
                (c*cell_width + 10, r*cell_height + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # Display the grid
        cv2.imshow("Preprocessing Examples", grid)
        
        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save individual samples
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            for title, proc_frame in processed_frames.items():
                filename = f"{output_dir}/{title}_{timestamp}.jpg"
                cv2.imwrite(filename, proc_frame)
            print(f"Saved preprocessing samples to {output_dir}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Preprocessing test completed") 