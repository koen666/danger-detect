import cv2
import numpy as np
import argparse
import os
import time
from datetime import datetime
from models.video_capture import VideoCaptureManager
from utils.frame_validation import validate_frame, save_frame_sample, draw_diagnostics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advanced video frame analysis')
    parser.add_argument(
        '--source', 
        type=str, 
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='frame_analysis',
        help='Directory to save analysis results'
    )
    parser.add_argument(
        '--show_histograms', 
        action='store_true',
        help='Show histograms of color channels'
    )
    parser.add_argument(
        '--sample_interval', 
        type=int, 
        default=50,
        help='Save a sample frame every N frames'
    )
    parser.add_argument(
        '--max_frames', 
        type=int, 
        default=0,
        help='Maximum number of frames to analyze (0 = unlimited)'
    )
    return parser.parse_args()


def calculate_frame_quality(frame):
    """
    Calculate quality metrics for a frame.
    
    Args:
        frame: OpenCV frame
        
    Returns:
        dict: Quality metrics
    """
    if frame is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Calculate basic statistics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Calculate histogram of grayscale image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / np.sum(hist)
    
    # Calculate entropy (information content)
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-7))
    
    # Calculate sharpness using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'entropy': entropy,
        'sharpness': sharpness
    }


def draw_histograms(frame, window_name="Histograms"):
    """
    Draw histograms for each color channel of the frame.
    
    Args:
        frame: OpenCV frame
        window_name: Name of the window to display histograms
        
    Returns:
        histogram_image: Image with histograms
    """
    if frame is None:
        return None
    
    # Create a blank image for the histograms
    hist_h = 256
    hist_w = 512
    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    
    # Calculate histograms for each channel
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        hist = (hist / hist.max() * hist_h * 0.9).astype(int)
        
        # Draw histogram
        for x in range(256):
            y = hist_h - hist[x]
            cv2.line(
                hist_image,
                (x * 2, hist_h),
                (x * 2, y),
                (255 if col == 'b' else 0, 255 if col == 'g' else 0, 255 if col == 'r' else 0),
                2
            )
    
    # Draw grid lines and labels
    for y in range(0, hist_h, 64):
        cv2.line(hist_image, (0, y), (hist_w, y), (40, 40, 40), 1)
        
    for x in range(0, 257, 64):
        cv2.line(hist_image, (x * 2, 0), (x * 2, hist_h), (40, 40, 40), 1)
        cv2.putText(
            hist_image,
            str(x),
            (x * 2 - 10, hist_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1
        )
    
    # Add title
    cv2.putText(
        hist_image,
        "RGB Histograms",
        (hist_w // 2 - 60, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return hist_image


def create_analysis_dashboard(frame, frame_num, diagnostics, quality_metrics):
    """
    Create a comprehensive analysis dashboard for a frame.
    
    Args:
        frame: Original frame
        frame_num: Frame number
        diagnostics: Frame diagnostics
        quality_metrics: Frame quality metrics
        
    Returns:
        dashboard: Dashboard image
    """
    if frame is None or diagnostics is None or quality_metrics is None:
        return None
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create a larger canvas for the dashboard
    dashboard_height = height + 250  # Add space for metrics
    dashboard = np.zeros((dashboard_height, width, 3), dtype=np.uint8)
    
    # Copy the original frame to the top of the dashboard
    dashboard[:height, :width] = frame
    
    # Add separating line
    cv2.line(dashboard, (0, height), (width, height), (100, 100, 100), 2)
    
    # Add frame metadata section
    text_y = height + 30
    
    # Frame number and dimensions
    cv2.putText(
        dashboard,
        f"Frame: {frame_num} | Dimensions: {width}x{height}",
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    text_y += 30
    
    # Frame validation status
    status_color = (0, 255, 0) if diagnostics['valid'] else (0, 0, 255)
    cv2.putText(
        dashboard,
        f"Validation: {'PASS' if diagnostics['valid'] else 'FAIL'}",
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2
    )
    text_y += 30
    
    # Quality metrics
    cv2.putText(
        dashboard,
        f"Brightness: {quality_metrics['brightness']:.1f} | Contrast: {quality_metrics['contrast']:.1f}",
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )
    text_y += 30
    
    cv2.putText(
        dashboard,
        f"Sharpness: {quality_metrics['sharpness']:.1f} | Entropy: {quality_metrics['entropy']:.1f}",
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )
    text_y += 30
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        dashboard,
        timestamp,
        (20, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1
    )
    
    return dashboard


def run_frame_analysis():
    """Run comprehensive frame analysis on video source."""
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
    
    # Get video properties
    video_props = capture_manager.get_video_properties()
    print(f"Video source opened successfully: {video_props['width']}x{video_props['height']} @ {video_props['fps']:.2f} fps")
    print("Press 'q' to exit, 's' to save current frame, 'p' to pause/resume")
    
    # Initialize analysis variables
    frames_processed = 0
    quality_metrics_history = []
    is_paused = False
    
    # Create a log file for metrics
    log_path = os.path.join(args.output_dir, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(log_path, 'w') as log_file:
        log_file.write("frame,timestamp,valid,brightness,contrast,entropy,sharpness\n")
    
        # Process frames
        for _, (success, frame, frame_num) in enumerate(capture_manager.read_frames(validate=True)):
            if not success:
                print("Failed to read frame. Exiting.")
                break
            
            # Handle pause state
            while is_paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):  # Resume on 'p' key
                    is_paused = False
                    print("Resuming playback")
                elif key == ord('q'):  # Quit on 'q' key even when paused
                    print("User requested exit while paused")
                    break
                elif key == ord('s'):  # Save current frame
                    save_path = save_frame_sample(frame, frame_num, args.output_dir)
                    print(f"Saved current frame to {save_path}")
            
            # Check if we've reached the maximum number of frames
            if args.max_frames > 0 and frames_processed >= args.max_frames:
                print(f"Reached maximum number of frames ({args.max_frames}). Exiting.")
                break
            
            # Validate the frame
            diagnostics = validate_frame(frame, frame_num)
            
            # Calculate quality metrics
            quality_metrics = calculate_frame_quality(frame)
            if quality_metrics:
                quality_metrics_history.append(quality_metrics)
            
            # Log metrics to file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{frame_num},{timestamp},{diagnostics['valid']},{quality_metrics['brightness']:.2f},"
                          f"{quality_metrics['contrast']:.2f},{quality_metrics['entropy']:.2f},{quality_metrics['sharpness']:.2f}\n")
            
            # Create analysis dashboard
            dashboard = create_analysis_dashboard(frame, frame_num, diagnostics, quality_metrics)
            
            # Display the dashboard
            cv2.imshow('Video Analysis', dashboard)
            
            # Show histograms if requested
            if args.show_histograms:
                hist_image = draw_histograms(frame)
                cv2.imshow('RGB Histograms', hist_image)
            
            # Save sample frames at regular intervals
            if frames_processed % args.sample_interval == 0:
                saved_path = save_frame_sample(frame, frame_num, args.output_dir)
                if saved_path:
                    print(f"Saved sample frame #{frame_num} to {saved_path}")
            
            # Track frames processed
            frames_processed += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested exit")
                break
            elif key == ord('s'):
                # Save current frame
                save_path = save_frame_sample(frame, frame_num, args.output_dir)
                print(f"Saved current frame to {save_path}")
            elif key == ord('p'):
                # Toggle pause
                is_paused = not is_paused
                print("Paused playback" if is_paused else "Resuming playback")
    
    # Get and print final statistics
    stats = capture_manager.get_processing_stats()
    print("\n--- Frame Analysis Results ---")
    print(f"Total frames analyzed: {frames_processed}")
    print(f"Average processing rate: {stats['processing_fps']:.2f} FPS")
    
    # Calculate and print quality metrics statistics
    if quality_metrics_history:
        avg_brightness = np.mean([qm['brightness'] for qm in quality_metrics_history])
        avg_contrast = np.mean([qm['contrast'] for qm in quality_metrics_history])
        avg_entropy = np.mean([qm['entropy'] for qm in quality_metrics_history])
        avg_sharpness = np.mean([qm['sharpness'] for qm in quality_metrics_history])
        
        print("\n--- Quality Metrics Summary ---")
        print(f"Average brightness: {avg_brightness:.2f}")
        print(f"Average contrast: {avg_contrast:.2f}")
        print(f"Average entropy: {avg_entropy:.2f}")
        print(f"Average sharpness: {avg_sharpness:.2f}")
    
    # Clean up
    capture_manager.close()
    cv2.destroyAllWindows()
    print(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    run_frame_analysis() 