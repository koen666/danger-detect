import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime
from models.video_capture import VideoCaptureManager
from models.qwen_vl import QwenVLFeatureExtractor
from models.frame_processor import FrameProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Qwen-VL Multimodal Features Test')
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source (camera index or file path)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='qwen_vl_results',
        help='Directory to save output results'
    )
    parser.add_argument(
        '--model_version',
        type=str,
        default='Qwen/Qwen-VL-Chat',
        help='Qwen-VL model version to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run the model on (cuda or cpu)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['caption', 'vqa', 'anomaly', 'classification'],
        default='caption',
        help='Operation mode'
    )
    parser.add_argument(
        '--question',
        type=str,
        default='What is happening in this scene?',
        help='Question for VQA mode'
    )
    parser.add_argument(
        '--normal_context',
        type=str,
        default='A normal scene with people walking peacefully.',
        help='Description of normal conditions for anomaly detection'
    )
    parser.add_argument(
        '--categories',
        type=str,
        default='indoor,outdoor,urban,rural,day,night,crowded,empty',
        help='Comma-separated categories for classification'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Apply preprocessing to frames'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Process every N-th frame'
    )
    return parser.parse_args()


class QwenVLDemo:
    """Interactive demo for Qwen-VL multimodal features."""
    
    def __init__(self, args):
        """
        Initialize the demo.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.source = int(args.source) if args.source.isdigit() else args.source
        self.output_dir = args.output_dir
        self.mode = args.mode
        self.interval = args.interval
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parse categories if in classification mode
        if self.mode == 'classification':
            self.categories = args.categories.split(',')
            print(f"Classification categories: {self.categories}")
        
        # Initialize components
        self.capture_manager = VideoCaptureManager()
        
        # Initialize frame processor if preprocessing is enabled
        self.use_preprocessing = args.preprocess
        if self.use_preprocessing:
            self.frame_processor = FrameProcessor({
                'resize_dim': None,  # Keep original size for VL model
                'normalize': False,  # Keep as uint8 for display
                'denoise': True,
                'denoise_method': 'gaussian',
                'enhance': True,
                'equalize_hist': False
            })
            print("Frame preprocessing enabled")
        
        # Initialize Qwen-VL feature extractor
        print(f"Initializing Qwen-VL model ({args.model_version})...")
        self.feature_extractor = QwenVLFeatureExtractor(
            model_version=args.model_version,
            device=args.device
        )
        
        if not self.feature_extractor.is_initialized:
            print("Failed to initialize Qwen-VL model")
            exit(1)
        
        # UI state
        self.is_paused = False
        self.show_help = False
        self.frame_count = 0
        self.last_processed_frame = None
        self.last_result = None
        self.processing_time = 0
        self.is_processing = False
    
    def connect_video_source(self):
        """Connect to the video source."""
        print(f"Connecting to video source: {self.source}")
        return self.capture_manager.open_source(self.source)
    
    def process_frame(self, frame):
        """
        Process a frame using Qwen-VL.
        
        Args:
            frame: Frame to process
            
        Returns:
            Processing result
        """
        # Apply preprocessing if enabled
        if self.use_preprocessing:
            processed_frame = self.frame_processor.process_frame(frame)
        else:
            processed_frame = frame
        
        start_time = time.time()
        
        # Process according to mode
        if self.mode == 'caption':
            result = self.feature_extractor.generate_caption(processed_frame)
        
        elif self.mode == 'vqa':
            result = self.feature_extractor.answer_question(processed_frame, self.args.question)
        
        elif self.mode == 'anomaly':
            anomaly_result = self.feature_extractor.detect_anomalies(
                processed_frame, self.args.normal_context
            )
            result = anomaly_result
        
        elif self.mode == 'classification':
            result = self.feature_extractor.classify_scene(processed_frame, self.categories)
        
        else:
            # Default to feature extraction
            features = self.feature_extractor.extract_features(processed_frame)
            if features is not None:
                result = f"Feature vector shape: {features.shape}"
            else:
                result = "Failed to extract features"
        
        # Calculate processing time
        self.processing_time = time.time() - start_time
        
        return result
    
    def run_demo(self):
        """Run the Qwen-VL demo."""
        if not self.connect_video_source():
            print("Failed to connect to video source. Exiting.")
            return
        
        print(f"\nRunning in {self.mode} mode")
        print("Controls:")
        print("  'q': Quit")
        print("  'p': Pause/Resume")
        print("  'h': Toggle help overlay")
        print("  's': Save current result")
        print("  'space': Process current frame")
        
        # Main loop
        for _, (success, frame, frame_num) in enumerate(self.capture_manager.read_frames(validate=False)):
            if not success:
                print("Failed to read frame. Exiting.")
                break
            
            self.frame_count += 1
            
            # Create a copy of the frame for display
            display_frame = frame.copy()
            
            # Handle pause state
            if self.is_paused:
                key = self.handle_keypress(cv2.waitKey(50))
                if key == 27:  # ESC or 'q' to exit
                    break
                
                # Show the last processed result
                if self.last_result is not None:
                    self.draw_result(display_frame, self.last_result)
                
                # Display the frame
                cv2.imshow('Qwen-VL Demo', display_frame)
                continue
            
            # Process frames at specified interval or when space is pressed
            process_this_frame = (self.frame_count % self.interval == 0) and not self.is_processing
            
            if process_this_frame:
                self.is_processing = True
                print(f"Processing frame #{frame_num}...")
                
                # Store a copy of the frame for saving
                self.last_processed_frame = frame.copy()
                
                # Process the frame
                try:
                    self.last_result = self.process_frame(frame)
                    print(f"Result: {self.last_result}")
                    print(f"Processing time: {self.processing_time:.2f} seconds")
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    self.last_result = f"Error: {str(e)}"
                
                self.is_processing = False
            
            # Draw result on display frame
            if self.last_result is not None:
                self.draw_result(display_frame, self.last_result)
            
            # Display the frame
            cv2.imshow('Qwen-VL Demo', display_frame)
            
            # Handle keypress
            key = self.handle_keypress(cv2.waitKey(1))
            if key == 27:  # ESC or 'q' to exit
                break
        
        # Clean up
        self.cleanup()
    
    def draw_result(self, frame, result):
        """
        Draw the processing result on the frame.
        
        Args:
            frame: Frame to draw on
            result: Processing result
        """
        h, w = frame.shape[:2]
        
        # Create a semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 150), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add mode and processing time
        cv2.putText(
            frame, 
            f"Mode: {self.mode.upper()} | Time: {self.processing_time:.2f}s", 
            (10, h - 120), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            1
        )
        
        # Draw result based on mode
        if self.mode == 'caption' or self.mode == 'vqa':
            # For textual results, wrap the text
            if isinstance(result, str):
                lines = self.wrap_text(result, w - 20, font_scale=0.6)
                for i, line in enumerate(lines):
                    y = h - 90 + i * 25
                    if y < h - 10:  # Prevent drawing outside frame
                        cv2.putText(
                            frame, line, (10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                        )
        
        elif self.mode == 'anomaly':
            # For anomaly detection
            if isinstance(result, dict):
                has_anomalies = result.get('has_anomalies', False)
                description = result.get('description', '')
                
                # Draw status
                status_color = (0, 0, 255) if has_anomalies else (0, 255, 0)
                status_text = "ANOMALY DETECTED" if has_anomalies else "Normal"
                cv2.putText(
                    frame, status_text, (10, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2
                )
                
                # Draw description
                lines = self.wrap_text(description, w - 20, font_scale=0.6)
                for i, line in enumerate(lines[:3]):  # Limit to 3 lines
                    y = h - 60 + i * 25
                    if y < h - 10:
                        cv2.putText(
                            frame, line, (10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                        )
        
        elif self.mode == 'classification':
            # For classification
            if isinstance(result, dict):
                category = result.get('category', 'Unknown')
                score = result.get('score', 0.0)
                raw = result.get('raw_response', '')
                
                # Draw classification result
                cv2.putText(
                    frame, f"Category: {category} ({score:.2f})", 
                    (10, h - 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )
                
                # Draw raw response
                lines = self.wrap_text(raw, w - 20, font_scale=0.6)
                for i, line in enumerate(lines[:2]):  # Limit to 2 lines
                    y = h - 60 + i * 25
                    if y < h - 10:
                        cv2.putText(
                            frame, line, (10, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                        )
        
        # Processing indicator
        if self.is_processing:
            cv2.putText(
                frame,
                "PROCESSING...",
                (w - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # Add help overlay if enabled
        if self.show_help:
            self.add_help_overlay(frame)
    
    def wrap_text(self, text, max_width, font_scale=0.6, thickness=1):
        """
        Wrap text to fit within a given width.
        
        Args:
            text: Text to wrap
            max_width: Maximum width in pixels
            font_scale: Font scale
            thickness: Line thickness
            
        Returns:
            List of wrapped text lines
        """
        if not text:
            return []
            
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Try adding the word to the current line
            test_line = ' '.join(current_line + [word])
            size = cv2.getTextSize(
                test_line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )[0]
            
            # If it fits, add it to the current line
            if size[0] <= max_width:
                current_line.append(word)
            # Otherwise, start a new line
            else:
                if current_line:  # Avoid empty lines
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def add_help_overlay(self, frame):
        """
        Add a help overlay to the frame.
        
        Args:
            frame: Frame to draw on
        """
        h, w = frame.shape[:2]
        
        # Create a semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Add help text
        help_text = [
            "CONTROLS:",
            "q/ESC: Quit",
            "p: Pause/Resume",
            "h: Toggle help",
            "s: Save current result",
            "space: Process current frame",
            f"Mode: {self.mode.upper()}",
            f"Processing time: {self.processing_time:.2f}s"
        ]
        
        for i, text in enumerate(help_text):
            cv2.putText(
                frame, 
                text, 
                (20, 40 + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                1
            )
    
    def save_current_result(self):
        """Save the current frame and result."""
        if self.last_processed_frame is None or self.last_result is None:
            print("No processed frame or result to save")
            return
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save frame
        frame_path = f"{self.output_dir}/frame_{timestamp}.jpg"
        cv2.imwrite(frame_path, self.last_processed_frame)
        
        # Save result to text file
        result_path = f"{self.output_dir}/result_{timestamp}.txt"
        with open(result_path, 'w') as f:
            f.write(f"Mode: {self.mode}\n")
            f.write(f"Processing time: {self.processing_time:.2f}s\n\n")
            
            if isinstance(self.last_result, dict):
                for k, v in self.last_result.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(self.last_result))
        
        print(f"Saved frame to {frame_path}")
        print(f"Saved result to {result_path}")
    
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
        
        key &= 0xFF
        
        if key == ord('q') or key == 27:  # q or ESC
            return 27
        
        elif key == ord('p'):  # Pause/Resume
            self.is_paused = not self.is_paused
            print("Playback " + ("paused" if self.is_paused else "resumed"))
        
        elif key == ord('h'):  # Toggle help
            self.show_help = not self.show_help
        
        elif key == ord('s'):  # Save result
            self.save_current_result()
        
        elif key == 32:  # Space - process current frame
            if not self.is_processing and not self.is_paused:
                # Get current frame
                _, frame, _ = next(self.capture_manager.read_frames())
                self.last_processed_frame = frame.copy()
                
                # Process the frame
                print("Processing current frame...")
                self.is_processing = True
                try:
                    self.last_result = self.process_frame(frame)
                    print(f"Result: {self.last_result}")
                    print(f"Processing time: {self.processing_time:.2f} seconds")
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
                    self.last_result = f"Error: {str(e)}"
                self.is_processing = False
        
        return key
    
    def cleanup(self):
        """Clean up resources."""
        self.capture_manager.close()
        cv2.destroyAllWindows()


def main():
    """Main entry point for the demo."""
    args = parse_args()
    demo = QwenVLDemo(args)
    demo.run_demo()


if __name__ == "__main__":
    main() 