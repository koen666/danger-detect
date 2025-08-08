import cv2
import os
import time
import argparse
from models.video_capture import VideoCaptureManager
from utils.frame_validation import validate_frame, save_frame_sample, draw_diagnostics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test video frame validation')
    parser.add_argument(
        '--source', 
        type=str, 
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--save_interval', 
        type=int, 
        default=100,
        help='Save a sample frame every N frames'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='frame_samples',
        help='Directory to save sample frames'
    )
    parser.add_argument(
        '--max_frames', 
        type=int, 
        default=0,
        help='Maximum number of frames to process (0 = unlimited)'
    )
    return parser.parse_args()


def test_frame_validation():
    """Test video frame capture and validation."""
    # Parse command-line arguments
    args = parse_args()
    
    # Convert source to int if it's a number (camera index)
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    # Create video capture manager
    capture_manager = VideoCaptureManager()
    
    print(f"Attempting to open video source: {source}")
    
    if not capture_manager.open_source(source):
        print("Failed to open video source. Exiting.")
        return
    
    print("Video source opened successfully. Press 'q' to exit.")
    
    # Initialize statistics
    start_time = time.time()
    frames_processed = 0
    valid_frames = 0
    invalid_frames = 0
    
    # Process frames
    for _, (success, frame, frame_num) in enumerate(capture_manager.read_frames()):
        if not success:
            print("Failed to read frame. Exiting.")
            break
        
        # Check if we've reached the maximum number of frames
        if args.max_frames > 0 and frames_processed >= args.max_frames:
            print(f"Reached maximum number of frames ({args.max_frames}). Exiting.")
            break
        
        # Validate the frame
        diagnostics = validate_frame(frame, frame_num)
        frames_processed += 1
        
        if diagnostics['valid']:
            valid_frames += 1
        else:
            invalid_frames += 1
            print(f"Invalid frame detected: #{frame_num}")
        
        # Save sample frames at regular intervals
        if frames_processed % args.save_interval == 0:
            saved_path = save_frame_sample(frame, frame_num, args.output_dir)
            if saved_path:
                print(f"Saved sample frame #{frame_num} to {saved_path}")
        
        # Draw diagnostics on frame
        annotated_frame = draw_diagnostics(frame, diagnostics)
        
        # Display the annotated frame
        cv2.imshow('Frame Validation', annotated_frame)
        
        # Calculate overall statistics periodically
        if frames_processed % 30 == 0:
            elapsed = time.time() - start_time
            fps = frames_processed / elapsed if elapsed > 0 else 0
            
            print(f"Processed {frames_processed} frames ({fps:.2f} FPS)")
            print(f"Valid: {valid_frames}, Invalid: {invalid_frames}")
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit")
            break
    
    # Final statistics
    elapsed = time.time() - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0
    
    print("\n--- Final Statistics ---")
    print(f"Total frames processed: {frames_processed}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Valid frames: {valid_frames} ({valid_frames/frames_processed*100:.1f}%)")
    print(f"Invalid frames: {invalid_frames} ({invalid_frames/frames_processed*100:.1f}%)")
    
    # Clean up
    capture_manager.close()
    cv2.destroyAllWindows()
    print("Test completed")


if __name__ == "__main__":
    test_frame_validation() 