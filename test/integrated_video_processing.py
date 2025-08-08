import cv2
import numpy as np
import argparse
import time
import os
import threading
import queue
from datetime import datetime
from models.video_capture import VideoCaptureManager
from models.frame_processor import FrameProcessor
from models.qwen_vl import QwenVLFeatureExtractor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Integrated Video Processing Demo')
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Processing width'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Processing height'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output_frames',
        help='Directory to save output frames'
    )
    parser.add_argument(
        '--denoise',
        action='store_true',
        help='Apply denoising'
    )
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Apply contrast enhancement'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize pixel values'
    )
    parser.add_argument(
        '--equalize',
        action='store_true',
        help='Apply histogram equalization'
    )
    # Add Qwen-VL arguments
    parser.add_argument(
        '--enable_vl',
        action='store_true',
        help='Enable Qwen-VL features'
    )
    parser.add_argument(
        '--vl_mode',
        type=str,
        choices=['caption', 'vqa', 'anomaly', 'classification'],
        default='caption',
        help='Qwen-VL operating mode'
    )
    parser.add_argument(
        '--vl_model',
        type=str,
        default='Qwen/Qwen-VL-Chat',
        help='Qwen-VL model version'
    )
    parser.add_argument(
        '--vl_interval',
        type=int,
        default=30,
        help='Process VL features every N frames'
    )
    return parser.parse_args()


class ProcessingApp:
    """Interactive application for video processing demonstration."""
    
    def __init__(self, args):
        """
        Initialize the application.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.source = int(args.source) if args.source.isdigit() else args.source
        self.output_dir = args.output_dir
        self.process_dimensions = (args.width, args.height)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processing components
        self.capture_manager = VideoCaptureManager()
        
        # Create initial preprocessing configuration
        self.preprocessing_config = {
            'resize_dim': self.process_dimensions,
            'normalize': args.normalize,
            'denoise': args.denoise,
            'denoise_method': 'gaussian',
            'enhance': args.enhance,
            'equalize_hist': args.equalize
        }
        
        # Initialize frame processor
        self.frame_processor = FrameProcessor(self.preprocessing_config)
        
        # Initialize Qwen-VL if enabled
        self.use_vl = args.enable_vl
        self.vl_mode = args.vl_mode
        self.vl_interval = args.vl_interval
        self.vl_extractor = None
        self.vl_result = None
        self.vl_processing_time = 0
        self.vl_frame_count = 0
        self.vl_processing = False
        self.vl_queue = queue.Queue(maxsize=1)  # Queue for VL processing
        self.vl_thread = None
        
        if self.use_vl:
            print(f"Initializing Qwen-VL model ({args.vl_model}) in {self.vl_mode} mode...")
            try:
                self.vl_extractor = QwenVLFeatureExtractor(
                    model_version=args.vl_model,
                    device=None  # Auto-detect device
                )
                if not self.vl_extractor.is_initialized:
                    print("Warning: Failed to initialize Qwen-VL model. Disabling VL features.")
                    self.use_vl = False
                else:
                    # Start VL processing thread
                    self.vl_thread = threading.Thread(target=self._vl_processing_thread, daemon=True)
                    self.vl_thread.start()
            except Exception as e:
                print(f"Error initializing Qwen-VL: {str(e)}")
                self.use_vl = False
        
        # User interface state
        self.is_paused = False
        self.show_preprocessing = True
        self.show_help = False
        self.recording = False
        self.record_frames = []
        self.max_record_frames = 100  # Maximum frames to record in memory
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_times = 30  # For calculating rolling average FPS
        self.last_frame_time = time.time()
    
    def connect_video_source(self):
        """Connect to the video source."""
        print(f"Connecting to video source: {self.source}")
        return self.capture_manager.open_source(self.source)
    
    def _vl_processing_thread(self):
        """Background thread for processing frames with Qwen-VL."""
        while True:
            try:
                # Get frame from queue
                frame = self.vl_queue.get()
                if frame is None:  # Signal to exit
                    break
                
                # Process the frame
                self.vl_processing = True
                start_time = time.time()
                
                try:
                    # Process according to mode
                    if self.vl_mode == 'caption':
                        result = self.vl_extractor.generate_caption(frame)
                    
                    elif self.vl_mode == 'vqa':
                        result = self.vl_extractor.answer_question(frame, "What is happening in this scene?")
                    
                    elif self.vl_mode == 'anomaly':
                        result = self.vl_extractor.detect_anomalies(
                            frame, "A normal scene with people walking peacefully."
                        )
                    
                    elif self.vl_mode == 'classification':
                        categories = ['indoor', 'outdoor', 'urban', 'rural', 'day', 'night', 'crowded', 'empty']
                        result = self.vl_extractor.classify_scene(frame, categories)
                    
                    else:
                        # Default to feature extraction
                        features = self.vl_extractor.extract_features(frame)
                        if features is not None:
                            result = f"Feature vector shape: {features.shape}"
                        else:
                            result = "Failed to extract features"
                    
                    # Update result and processing time
                    self.vl_result = result
                    self.vl_processing_time = time.time() - start_time
                    
                except Exception as e:
                    print(f"Error in VL processing: {str(e)}")
                    self.vl_result = f"Error: {str(e)}"
                    self.vl_processing_time = time.time() - start_time
                
                finally:
                    self.vl_processing = False
                    self.vl_queue.task_done()
            
            except Exception as e:
                print(f"Error in VL thread: {str(e)}")
    
    def process_video(self):
        """Main processing loop."""
        if not self.connect_video_source():
            print("Failed to connect to video source. Exiting.")
            return
        
        # Get video properties
        video_props = self.capture_manager.get_video_properties()
        print(f"Video source: {video_props['width']}x{video_props['height']} @ {video_props['fps']:.2f} fps")
        
        print("\nControls:")
        print("  'q': Quit")
        print("  'p': Pause/Resume")
        print("  'h': Toggle help overlay")
        print("  's': Save current frame")
        print("  'd': Toggle denoising")
        print("  'e': Toggle enhancement")
        print("  'n': Toggle normalization")
        print("  'r': Start/stop recording (max 100 frames)")
        print("  'o': Toggle preprocessing on/off")
        print("  '1-3': Change denoising method (1=gaussian, 2=bilateral, 3=nlmeans)")
        if self.use_vl:
            print("  'v': Force VL processing on current frame")
        
        # Main processing loop
        for _, (success, frame, frame_num) in enumerate(self.capture_manager.read_frames(validate=False)):
            if not success:
                print("Failed to read frame. Exiting.")
                break
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.frame_times.append(elapsed)
            self.frame_times = self.frame_times[-self.max_frame_times:]  # Keep only recent times
            
            # Handle pause state
            if self.is_paused:
                key = self.handle_keypress(cv2.waitKey(50))
                if key == 27:  # ESC or 'q' to exit
                    break
                continue
            
            # Apply preprocessing if enabled
            if self.show_preprocessing:
                processed_frame = self.frame_processor.process_frame(frame, frame_id=frame_num)
            else:
                processed_frame = frame.copy()
            
            # Process with Qwen-VL at intervals if enabled
            if self.use_vl:
                self.vl_frame_count += 1
                if self.vl_frame_count % self.vl_interval == 0 and not self.vl_processing and self.vl_queue.empty():
                    self.vl_queue.put(frame.copy())
            
            # Record frame if recording
            if self.recording:
                # Convert to uint8 if normalized
                if processed_frame.dtype == np.float32 or processed_frame.dtype == np.float64:
                    record_frame = (processed_frame * 255).astype(np.uint8)
                else:
                    record_frame = processed_frame.copy()
                
                self.record_frames.append(record_frame)
                # Limit the number of frames to avoid memory issues
                if len(self.record_frames) > self.max_record_frames:
                    self.recording = False
                    print(f"Stopped recording after {self.max_record_frames} frames")
            
            # Create display output
            display_frame = self.create_display(frame, processed_frame, frame_num)
            
            # Show the output
            cv2.imshow('Integrated Video Processing', display_frame)
            
            # Handle keypress
            key = self.handle_keypress(cv2.waitKey(1))
            if key == 27:  # ESC or 'q' to exit
                break
        
        # Clean up
        self.cleanup()
    
    def create_display(self, original, processed, frame_num):
        """
        Create a display frame with original, processed, and UI elements.
        
        Args:
            original: Original input frame
            processed: Processed output frame
            frame_num: Current frame number
            
        Returns:
            Combined display frame
        """
        h, w = original.shape[:2]
        
        # Resize original and processed frames if they're not already at display size
        if original.shape[:2] != self.process_dimensions:
            original_display = cv2.resize(original, self.process_dimensions)
        else:
            original_display = original.copy()
            
        if processed.shape[:2] != self.process_dimensions:
            processed_display = cv2.resize(processed, self.process_dimensions)
        else:
            processed_display = processed.copy()
            
        # Ensure processed frame is uint8 for display
        if processed_display.dtype != np.uint8:
            processed_display = (processed_display * 255).astype(np.uint8)
        
        # Create side-by-side display
        display_w = self.process_dimensions[0] * 2
        display_h = self.process_dimensions[1]
        display = np.zeros((display_h, display_w, 3), dtype=np.uint8)
        
        # Place original and processed frames
        display[:, :self.process_dimensions[0]] = original_display
        display[:, self.process_dimensions[0]:] = processed_display
        
        # Add labels
        cv2.putText(
            display, 
            "Original", 
            (10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        cv2.putText(
            display, 
            "Processed", 
            (self.process_dimensions[0] + 10, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        # Calculate and show FPS
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            cv2.putText(
                display, 
                f"FPS: {fps:.1f}", 
                (display_w - 100, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
        
        # Add frame number
        cv2.putText(
            display, 
            f"Frame: {frame_num}", 
            (display_w - 100, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        # Add preprocessing configuration text
        preproc_text = []
        if self.preprocessing_config['denoise']:
            preproc_text.append(f"Denoise: {self.preprocessing_config['denoise_method']}")
        if self.preprocessing_config['enhance']:
            preproc_text.append("Enhance")
        if self.preprocessing_config['normalize']:
            preproc_text.append("Normalize")
        if self.preprocessing_config['equalize_hist']:
            preproc_text.append("Hist. Eq.")
            
        if preproc_text:
            config_str = "Config: " + ", ".join(preproc_text)
        else:
            config_str = "Config: None"
            
        cv2.putText(
            display, 
            config_str, 
            (10, display_h - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        # Add recording indicator
        if self.recording:
            cv2.putText(
                display, 
                f"RECORDING ({len(self.record_frames)}/{self.max_record_frames})", 
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                1
            )
        
        # Add Qwen-VL results if available
        if self.use_vl and self.vl_result is not None:
            # Add a semi-transparent overlay at the bottom
            overlay = display.copy()
            cv2.rectangle(overlay, (0, display_h - 100), (display_w, display_h), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            
            # Add VL mode and processing time
            cv2.putText(
                display,
                f"Qwen-VL ({self.vl_mode.upper()}) | Time: {self.vl_processing_time:.2f}s", 
                (10, display_h - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Display result based on mode
            if self.vl_mode == 'caption' or self.vl_mode == 'vqa':
                if isinstance(self.vl_result, str):
                    if len(self.vl_result) > 60:
                        line1 = self.vl_result[:60]
                        line2 = self.vl_result[60:120] + ("..." if len(self.vl_result) > 120 else "")
                        cv2.putText(
                            display, 
                            line1, 
                            (10, display_h - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            1
                        )
                        cv2.putText(
                            display, 
                            line2, 
                            (10, display_h - 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            1
                        )
                    else:
                        cv2.putText(
                            display, 
                            self.vl_result, 
                            (10, display_h - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            1
                        )
            
            elif self.vl_mode == 'anomaly':
                if isinstance(self.vl_result, dict):
                    has_anomalies = self.vl_result.get('has_anomalies', False)
                    description = self.vl_result.get('description', '')
                    
                    # Draw status
                    status_color = (0, 0, 255) if has_anomalies else (0, 255, 0)
                    status_text = "ANOMALY DETECTED" if has_anomalies else "Normal"
                    cv2.putText(
                        display, 
                        status_text, 
                        (10, display_h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        status_color, 
                        2
                    )
                    
                    # Show description (limited)
                    if len(description) > 60:
                        description = description[:57] + "..."
                    cv2.putText(
                        display, 
                        description, 
                        (10, display_h - 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )
            
            elif self.vl_mode == 'classification':
                if isinstance(self.vl_result, dict):
                    category = self.vl_result.get('category', 'Unknown')
                    score = self.vl_result.get('score', 0.0)
                    
                    # Draw classification result
                    cv2.putText(
                        display, 
                        f"Category: {category} ({score:.2f})", 
                        (10, display_h - 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 255), 
                        1
                    )
        
        # Add help overlay if requested
        if self.show_help:
            # Create semi-transparent overlay
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display_w, display_h), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            
            # Add help text
            help_lines = [
                "KEYBOARD CONTROLS:",
                "q/ESC: Quit",
                "p: Pause/Resume",
                "h: Toggle help overlay",
                "s: Save current frame",
                "d: Toggle denoising",
                "e: Toggle enhancement",
                "n: Toggle normalization",
                "1-3: Change denoising method",
                "  1: Gaussian",
                "  2: Bilateral",
                "  3: Non-local means",
                "o: Toggle preprocessing on/off",
                "r: Start/stop recording (max 100 frames)"
            ]
            
            if self.use_vl:
                help_lines.append("v: Force VL processing on current frame")
            
            y_offset = 30
            for i, line in enumerate(help_lines):
                cv2.putText(
                    display,
                    line,
                    (20, y_offset + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return display
    
    def handle_keypress(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code
            
        Returns:
            27 to exit, otherwise the key code
        """
        if key == -1:  # No key pressed
            return key
        
        if key == ord('q') or key == 27:  # q or ESC
            return 27
        
        elif key == ord('p'):  # Pause/Resume
            self.is_paused = not self.is_paused
            print("Playback " + ("paused" if self.is_paused else "resumed"))
        
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
        
        elif key == ord('s'):  # Save current frame
            success, frame, _ = next(self.capture_manager.read_frames())
            if success:
                if self.show_preprocessing:
                    processed = self.frame_processor.process_frame(frame)
                else:
                    processed = frame
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")
                
                # Convert to uint8 if normalized
                if processed.dtype == np.float32 or processed.dtype == np.float64:
                    processed = (processed * 255).astype(np.uint8)
                
                cv2.imwrite(filename, processed)
                print(f"Saved frame to {filename}")
        
        elif key == ord('d'):  # Toggle denoising
            self.preprocessing_config['denoise'] = not self.preprocessing_config['denoise']
            print("Denoising " + ("enabled" if self.preprocessing_config['denoise'] else "disabled"))
        
        elif key == ord('e'):  # Toggle enhancement
            self.preprocessing_config['enhance'] = not self.preprocessing_config['enhance']
            print("Enhancement " + ("enabled" if self.preprocessing_config['enhance'] else "disabled"))
        
        elif key == ord('n'):  # Toggle normalization
            self.preprocessing_config['normalize'] = not self.preprocessing_config['normalize']
            print("Normalization " + ("enabled" if self.preprocessing_config['normalize'] else "disabled"))
        
        elif key == ord('o'):  # Toggle preprocessing
            self.show_preprocessing = not self.show_preprocessing
            print("Preprocessing " + ("enabled" if self.show_preprocessing else "disabled"))
        
        elif key == ord('r'):  # Start/stop recording
            if not self.recording:
                self.recording = True
                self.record_frames = []
                print("Started recording")
            else:
                self.recording = False
                print(f"Stopped recording. Captured {len(self.record_frames)} frames.")
                
                # Save frames
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_dir = os.path.join(self.output_dir, f"record_{timestamp}")
                os.makedirs(record_dir, exist_ok=True)
                
                print(f"Saving frames to {record_dir}")
                for i, frame in enumerate(self.record_frames):
                    frame_path = os.path.join(record_dir, f"frame_{i:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                self.record_frames = []
        
        elif key in [ord('1'), ord('2'), ord('3')]:  # Change denoising method
            if key == ord('1'):
                self.preprocessing_config['denoise_method'] = 'gaussian'
                print("Denoising method: Gaussian")
            elif key == ord('2'):
                self.preprocessing_config['denoise_method'] = 'bilateral'
                print("Denoising method: Bilateral")
            elif key == ord('3'):
                self.preprocessing_config['denoise_method'] = 'nlm'
                print("Denoising method: Non-local Means")
                
        elif key == ord('v') and self.use_vl:  # Force VL processing
            if not self.vl_processing and self.vl_queue.empty():
                # Get current frame
                success, frame, _ = next(self.capture_manager.read_frames())
                if success:
                    print("Processing frame with Qwen-VL...")
                    self.vl_queue.put(frame.copy())
        
        return key
    
    def cleanup(self):
        """Clean up resources."""
        # Save recorded frames if any
        if self.record_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_dir = os.path.join(self.output_dir, f"record_{timestamp}")
            os.makedirs(record_dir, exist_ok=True)
            
            print(f"Saving {len(self.record_frames)} recorded frames to {record_dir}")
            for i, frame in enumerate(self.record_frames):
                frame_path = os.path.join(record_dir, f"frame_{i:04d}.jpg")
                cv2.imwrite(frame_path, frame)
        
        # Stop VL thread if it exists
        if self.use_vl and self.vl_thread is not None and self.vl_thread.is_alive():
            # Signal thread to exit and wait for it to finish
            try:
                self.vl_queue.put(None)
                self.vl_thread.join(timeout=1.0)
                print("VL processing thread stopped")
            except Exception as e:
                print(f"Error stopping VL thread: {str(e)}")
        
        # Close video capture
        self.capture_manager.close()
        
        # Close all windows
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main entry point for the application."""
    args = parse_args()
    app = ProcessingApp(args)
    app.process_video()


if __name__ == "__main__":
    main() 