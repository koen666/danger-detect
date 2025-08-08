import cv2
import numpy as np
import os
import argparse
import time
from models.video_capture import VideoCaptureManager
from utils.preprocessing import (
    preprocess_frame, apply_denoising, enhance_frame, 
    equalize_histogram, edge_detection, normalize_frame
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test video frame preprocessing')
    parser.add_argument(
        '--source', 
        type=str, 
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='preprocessing_samples',
        help='Directory to save sample frames'
    )
    parser.add_argument(
        '--width', 
        type=int, 
        default=640,
        help='Width to resize frames'
    )
    parser.add_argument(
        '--height', 
        type=int, 
        default=480,
        help='Height to resize frames'
    )
    parser.add_argument(
        '--save_interval', 
        type=int, 
        default=30,
        help='Save a sample frame every N frames'
    )
    parser.add_argument(
        '--denoise_method', 
        type=str, 
        default='gaussian',
        choices=['gaussian', 'bilateral', 'nlmeans'],
        help='Denoising method to use'
    )
    return parser.parse_args()


def create_processing_grid(frames_dict, grid_size=(2, 3), cell_size=(320, 240)):
    """
    Create a grid visualization of multiple processed frames.
    
    Args:
        frames_dict: Dictionary of {title: frame}
        grid_size: Tuple (rows, cols) for grid layout
        cell_size: Tuple (width, height) for each cell
        
    Returns:
        Grid image combining all frames
    """
    rows, cols = grid_size
    cell_width, cell_height = cell_size
    
    # Create empty grid
    grid = np.zeros((rows * cell_height, cols * cell_width, 3), dtype=np.uint8)
    
    # Place frames in grid
    for i, (title, frame) in enumerate(frames_dict.items()):
        if i >= rows * cols:  # Skip if more frames than grid cells
            break
            
        r, c = i // cols, i % cols
        
        # Ensure frame is 3-channel for display
        if len(frame.shape) == 2:  # Grayscale
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.dtype == np.float32 or frame.dtype == np.float64:
            # Convert normalized float frames back to uint8 for display
            display_frame = (frame * 255).astype(np.uint8)
        else:
            display_frame = frame.copy()
        
        # Resize for display
        display_frame = cv2.resize(display_frame, (cell_width, cell_height))
        
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
    
    return grid


def test_preprocessing():
    """Test video frame preprocessing with visualization."""
    # Parse command-line arguments
    args = parse_args()
    
    # Convert source to int if it's a number (camera index)
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create video capture manager
    capture_manager = VideoCaptureManager()
    
    print(f"Attempting to open video source: {source}")
    
    if not capture_manager.open_source(source):
        print("Failed to open video source. Exiting.")
        return
    
    print("Video source opened successfully.")
    print("Press 'q' to exit, 's' to save current preprocessing grid, 'p' to pause/resume")
    
    # Initialize variables
    frame_count = 0
    is_paused = False
    target_size = (args.width, args.height)
    
    # Process frames
    for _, (success, frame, frame_num) in enumerate(capture_manager.read_frames(validate=False)):
        if not success:
            print("Failed to read frame. Exiting.")
            break
        
        # Handle pause state
        while is_paused:
            key = cv2.waitKey(100) & 0xFF
            if key == ord('p'):  # Resume on 'p' key
                is_paused = False
                print("Resuming playback")
                break
            elif key == ord('q'):  # Quit on 'q' key even when paused
                print("User requested exit while paused")
                return
            elif key == ord('s'):  # Save current grid when paused
                save_grid_samples(frame, args.output_dir, frame_num, target_size, args.denoise_method)
                print(f"Saved preprocessing samples while paused")
        
        # Process the frame in multiple ways
        processed_frames = process_frame_with_variations(frame, target_size, args.denoise_method)
        
        # Create visualization grid
        grid = create_processing_grid(processed_frames)
        
        # Display the grid
        cv2.imshow("Frame Preprocessing", grid)
        
        # Save samples at regular intervals
        frame_count += 1
        if frame_count % args.save_interval == 0:
            save_grid_samples(frame, args.output_dir, frame_num, target_size, args.denoise_method)
            print(f"Saved preprocessing samples at frame #{frame_num}")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User requested exit")
            break
        elif key == ord('s'):
            save_grid_samples(frame, args.output_dir, frame_num, target_size, args.denoise_method)
            print(f"Saved preprocessing samples at frame #{frame_num}")
        elif key == ord('p'):
            is_paused = not is_paused
            print("Paused playback" if is_paused else "Resuming playback")
    
    # Clean up
    capture_manager.close()
    cv2.destroyAllWindows()
    print("Preprocessing test completed")


def process_frame_with_variations(frame, target_size=(640, 480), denoise_method='gaussian'):
    """
    Process a frame with different preprocessing combinations.
    
    Args:
        frame: Input frame
        target_size: Size to resize frames to
        denoise_method: Method to use for denoising
        
    Returns:
        Dictionary of processed frames with titles
    """
    # Create a copy of the original frame
    original = frame.copy()
    
    # Resize to target size for consistent processing
    resized = cv2.resize(original, target_size)
    
    # Apply different preprocessing operations
    denoised = apply_denoising(resized, method=denoise_method)
    enhanced = enhance_frame(resized)
    equalized = equalize_histogram(resized)
    edges = edge_detection(resized)
    
    # Full preprocessing pipeline
    full_processed = preprocess_frame(
        resized, 
        resize_dim=target_size,
        normalize=False,  # Keep in uint8 for display
        denoise=True,
        enhance=True,
        equalize_hist=False
    )
    
    # Return dictionary of processed frames
    return {
        "Original": resized,
        "Denoised": denoised,
        "Enhanced": enhanced,
        "Equalized": equalized,
        "Edge Detection": cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
        "Full Pipeline": full_processed
    }


def save_grid_samples(frame, output_dir, frame_num, target_size, denoise_method):
    """
    Save all preprocessing variations of a frame.
    
    Args:
        frame: Input frame
        output_dir: Directory to save samples
        frame_num: Frame number for filename
        target_size: Size to resize frames to
        denoise_method: Method to use for denoising
    """
    # Get all preprocessing variations
    processed_frames = process_frame_with_variations(frame, target_size, denoise_method)
    
    # Create timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save each variation
    for title, proc_frame in processed_frames.items():
        # Convert normalized frames back to uint8 for saving
        if proc_frame.dtype == np.float32 or proc_frame.dtype == np.float64:
            save_frame = (proc_frame * 255).astype(np.uint8)
        else:
            save_frame = proc_frame
            
        filename = f"{output_dir}/{title}_frame{frame_num}_{timestamp}.jpg"
        cv2.imwrite(filename, save_frame)


if __name__ == "__main__":
    test_preprocessing() 