import cv2
import logging
import time
import numpy as np
from datetime import datetime


class VideoCaptureManager:
    """
    A class to manage video capture from different sources (camera or video file).
    """
    
    def __init__(self, source=0, width=640, height=480, fps=30, use_test_mode_on_failure=True):
        """
        Initialize the video capture manager.
        
        Args:
            source: Camera index (int) or video file path (str)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            use_test_mode_on_failure: If True, will use test pattern when source fails
        """
        self.cap = None
        self.source = source
        self.is_opened = False
        self.frame_count = 0
        
        # Video properties
        self.width = width
        self.height = height
        self.fps = fps
        self.target_fps = fps
        self.frame_count_total = 0
        
        # Test mode
        self.test_mode = False
        self.use_test_mode_on_failure = use_test_mode_on_failure
        self.test_pattern_counter = 0
        
        # Frame buffer
        self.current_frame = None
        self.previous_frame = None
        
        # Statistics
        self.start_time = 0
        self.processing_stats = {
            'frames_processed': 0,
            'valid_frames': 0,
            'invalid_frames': 0,
            'avg_brightness': 0,
            'avg_stddev': 0,
            'processing_fps': 0
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('VideoCaptureManager')
    
    def connect(self):
        """
        Connect to the video source.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = self.open_source(self.source)
        
        # If connection failed and test mode is enabled, use test mode
        if not success and self.use_test_mode_on_failure:
            self.logger.info("Source connection failed, using test pattern generator")
            self.test_mode = True
            self.is_opened = True
            self.frame_count_total = float('inf')  # Infinite frames for test pattern
            self.start_time = time.time()
            
            # Generate initial test frames
            self.current_frame = self._generate_test_pattern()
            self.previous_frame = self.current_frame.copy()
            self.logger.info(f"Test pattern mode activated: {self.width}x{self.height} @ {self.fps:.2f} fps")
            return True
            
        return success
    
    def open_source(self, source):
        """
        Open a video source for capture.
        
        Args:
            source: Camera index (int) or video file path (str)
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.source = source
        self.logger.info(f"Attempting to connect to video source: {source}")
        
        try:
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video source: {source}")
                return False
            
            # Set video properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if isinstance(source, int):  # Only set FPS for cameras
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Get actual video properties
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # For live cameras, frame count may be zero or negative
            if self.frame_count_total <= 0:
                self.frame_count_total = float('inf')  # Infinite frames for cameras
            
            self.logger.info(f"Video source connected successfully")
            self.logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps:.2f} fps")
            
            if self.frame_count_total != float('inf'):
                self.logger.info(f"Total frames in video: {self.frame_count_total}")
                self.logger.info(f"Estimated duration: {self.frame_count_total/self.fps:.2f} seconds")
            
            # Read first frame to confirm
            ret, first_frame = self.cap.read()
            if not ret or first_frame is None:
                self.logger.error("Failed to read first frame from video source")
                self.cap.release()
                return False
                
            self.logger.info(f"First frame read successfully, shape: {first_frame.shape}")
            
            # Initialize frame buffers
            self.current_frame = first_frame
            self.previous_frame = first_frame.copy()
            
            # Reset the video if it's a file (not a camera)
            if isinstance(source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.is_opened = True
            self.start_time = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening video source: {str(e)}")
            return False
    
    def _generate_test_pattern(self):
        """
        Generate a test pattern image.
        
        Returns:
            numpy.ndarray: Test pattern image
        """
        # Create a blank image
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Increment counter for animation
        self.test_pattern_counter += 1
        t = self.test_pattern_counter / 30.0  # Time variable for animation
        
        # Draw a grid
        cell_size = 50
        for y in range(0, self.height, cell_size):
            for x in range(0, self.width, cell_size):
                color = ((x + y) // cell_size) % 2 * 255
                cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), (color, color, color), -1)
        
        # Draw a moving circle
        radius = 50
        cx = int(self.width/2 + self.width/4 * np.sin(t))
        cy = int(self.height/2 + self.height/4 * np.cos(t))
        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)
        
        # Add some text
        cv2.putText(frame, f"TEST PATTERN - Time: {t:.1f}s", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def read(self):
        """
        Read a single frame from the video source.
        
        Returns:
            tuple: (success, frame)
        """
        if not self.is_opened:
            self.logger.error("Cannot read frame: No video source opened")
            return False, None
        
        try:
            # Save the current frame as previous
            if self.current_frame is not None:
                self.previous_frame = self.current_frame.copy()
            
            # In test mode, generate test pattern
            if self.test_mode:
                frame = self._generate_test_pattern()
                self.current_frame = frame
                self.frame_count += 1
                
                # Simulate target FPS
                target_time = self.frame_count / self.fps
                elapsed = time.time() - self.start_time
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
                
                # Update FPS statistics
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.start_time
                    self.processing_stats['processing_fps'] = self.frame_count / elapsed if elapsed > 0 else 0
                
                return True, frame
            
            # Normal mode using OpenCV
            if self.cap is None:
                self.logger.error("Cannot read frame: Video capture is not initialized")
                return False, None
                
            # Read new frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                self.logger.warning("End of video stream reached or frame reading failed")
                return False, None
            
            self.frame_count += 1
            self.current_frame = frame
            
            # Update FPS statistics
            if self.frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - self.start_time
                self.processing_stats['processing_fps'] = self.frame_count / elapsed if elapsed > 0 else 0
            
            return ret, frame
            
        except Exception as e:
            self.logger.error(f"Error during frame reading: {str(e)}")
            return False, None
    
    def read_frames(self, validate=True):
        """
        Generator function to continuously read frames from the video source.
        
        Args:
            validate: Whether to perform basic frame validation
        
        Yields:
            tuple: (success, frame, frame_number)
        """
        if not self.is_opened or self.cap is None:
            self.logger.error("Cannot read frames: No video source opened")
            return
        
        self.frame_count = 0
        start_time = time.time()
        frames_processed = 0
        
        # Reset statistics
        self.processing_stats = {
            'frames_processed': 0,
            'valid_frames': 0,
            'invalid_frames': 0,
            'avg_brightness': 0,
            'avg_stddev': 0,
            'processing_fps': 0
        }
        
        try:
            while self.is_opened:
                ret, frame = self.read()
                
                if not ret or frame is None:
                    self.logger.info("End of video stream reached or frame reading failed")
                    break
                
                frames_processed += 1
                self.processing_stats['frames_processed'] += 1
                
                # Basic frame validation
                if validate:
                    valid_frame = self._validate_frame(frame)
                    if valid_frame:
                        self.processing_stats['valid_frames'] += 1
                    else:
                        self.processing_stats['invalid_frames'] += 1
                        self.logger.warning(f"Invalid frame detected: #{self.frame_count}")
                
                # Calculate FPS every 30 frames
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frames_processed / elapsed if elapsed > 0 else 0
                    self.processing_stats['processing_fps'] = fps
                    
                    # Log progress
                    progress = ""
                    if self.frame_count_total != float('inf'):
                        progress = f" ({self.frame_count/self.frame_count_total*100:.1f}%)"
                    
                    self.logger.info(f"Processing at {fps:.2f} FPS, frame #{self.frame_count}{progress}")
                    
                    # Reset for next interval
                    start_time = current_time
                    frames_processed = 0
                
                yield ret, frame, self.frame_count
                
        except Exception as e:
            self.logger.error(f"Error during frame reading: {str(e)}")
        finally:
            self.logger.info("Exiting frame reading loop")
            
            # Log final statistics
            total_elapsed = time.time() - self.start_time
            if self.processing_stats['frames_processed'] > 0:
                avg_fps = self.processing_stats['frames_processed'] / total_elapsed if total_elapsed > 0 else 0
                self.logger.info(f"Processed {self.processing_stats['frames_processed']} frames at avg {avg_fps:.2f} FPS")
                
                # Log validation statistics if validation was enabled
                if validate:
                    valid_percent = self.processing_stats['valid_frames'] / self.processing_stats['frames_processed'] * 100
                    self.logger.info(f"Valid frames: {self.processing_stats['valid_frames']} ({valid_percent:.1f}%)")
                    self.logger.info(f"Invalid frames: {self.processing_stats['invalid_frames']}")
    
    def _validate_frame(self, frame):
        """
        Perform basic validation on a frame.
        
        Args:
            frame: OpenCV frame to validate
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False
        
        # Check for empty or corrupted frames
        if frame.size == 0:
            return False
        
        # Check for NaN values
        if np.isnan(frame).any():
            return False
        
        # Calculate brightness and standard deviation for content analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # Update running averages
        if self.processing_stats['frames_processed'] > 0:
            n = self.processing_stats['frames_processed']
            self.processing_stats['avg_brightness'] = (self.processing_stats['avg_brightness'] * (n - 1) + brightness) / n
            self.processing_stats['avg_stddev'] = (self.processing_stats['avg_stddev'] * (n - 1) + std_dev) / n
        else:
            self.processing_stats['avg_brightness'] = brightness
            self.processing_stats['avg_stddev'] = std_dev
        
        # Check if frame has content - not completely black or white
        has_content = std_dev > 5.0  # Threshold for determining if frame has content
        
        return has_content
    
    def get_previous_frame(self):
        """
        Get the previous frame captured.
        
        Returns:
            numpy.ndarray: Previous frame or None if not available
        """
        return self.previous_frame
    
    def get_frame_count(self):
        """Get the current frame count."""
        return self.frame_count
    
    def get_fps(self):
        """Get the current processing FPS."""
        return self.processing_stats['processing_fps']
    
    def get_video_properties(self):
        """
        Get the properties of the current video source.
        
        Returns:
            dict: Dictionary with video properties
        """
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count_total,
            'duration': self.frame_count_total / self.fps if self.fps > 0 else 0
        }
    
    def get_processing_stats(self):
        """
        Get the current processing statistics.
        
        Returns:
            dict: Dictionary with processing statistics
        """
        return self.processing_stats
    
    def release(self):
        """Release the video capture resources."""
        self.close()
    
    def close(self):
        """Close the video capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            self.logger.info("Video capture resources released")
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


# Example usage
if __name__ == "__main__":
    # Create instance
    capture_manager = VideoCaptureManager()
    
    # Open camera (camera index 0) or video file
    source = 0  # Use camera index 0 (default camera)
    # source = "path/to/video.mp4"  # Or use a video file
    
    if capture_manager.open_source(source):
        # Display first 100 frames (or until end of video)
        for i, (success, frame, frame_num) in enumerate(capture_manager.read_frames(validate=True)):
            if i >= 100:  # Limit to 100 frames for this example
                break
                
            # Display frame number on the frame
            cv2.putText(
                frame, 
                f"Frame: {frame_num}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display the frame
            cv2.imshow('Video Frame', frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Print final statistics
        stats = capture_manager.get_processing_stats()
        print(f"Processed {stats['frames_processed']} frames at {stats['processing_fps']:.2f} FPS")
        if stats['frames_processed'] > 0:
            valid_percent = stats['valid_frames'] / stats['frames_processed'] * 100
            print(f"Valid frames: {stats['valid_frames']} ({valid_percent:.1f}%)")
            print(f"Invalid frames: {stats['invalid_frames']}")
            print(f"Average brightness: {stats['avg_brightness']:.2f}")
            print(f"Average standard deviation: {stats['avg_stddev']:.2f}")
        
        # Clean up
        capture_manager.close()
        cv2.destroyAllWindows() 