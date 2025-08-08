import cv2
import logging
import time
import numpy as np
from utils.preprocessing import preprocess_frame


class FrameProcessor:
    """
    A class to handle frame processing pipelines, including preprocessing,
    feature extraction, and optional caching of results.
    """
    
    def __init__(self, preprocessing_config=None):
        """
        Initialize the frame processor with configuration.
        
        Args:
            preprocessing_config: Dictionary of preprocessing parameters
        """
        # Set default preprocessing configuration
        self.preprocessing_config = preprocessing_config or {
            'resize_dim': (640, 480),
            'normalize': True,
            'denoise': True,
            'denoise_method': 'gaussian',
            'enhance': True,
            'equalize_hist': False
        }
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FrameProcessor')
        
        # Statistics tracking
        self.processing_stats = {
            'frames_processed': 0,
            'processing_time': 0,
            'avg_processing_time': 0
        }
        
        # Frame cache (optional)
        self.frame_cache = {}
        self.max_cache_size = 30  # Maximum number of frames to cache
        
        self.logger.info("Frame processor initialized with configuration:")
        for key, value in self.preprocessing_config.items():
            self.logger.info(f"  {key}: {value}")
    
    def process_frame(self, frame, frame_id=None, cache_result=False):
        """
        Process a single frame through the preprocessing pipeline.
        
        Args:
            frame: Input frame to process
            frame_id: Optional identifier for the frame (for caching)
            cache_result: Whether to cache the processed frame
            
        Returns:
            Processed frame
        """
        if frame is None:
            self.logger.warning("Cannot process None frame")
            return None
        
        start_time = time.time()
        
        # Apply preprocessing
        processed = self._preprocess(frame)
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(processing_time)
        
        # Cache the result if requested and an ID is provided
        if cache_result and frame_id is not None:
            self._cache_frame(frame_id, processed)
        
        return processed
    
    def process_batch(self, frames, frame_ids=None):
        """
        Process a batch of frames.
        
        Args:
            frames: List of frames to process
            frame_ids: Optional list of frame identifiers
            
        Returns:
            List of processed frames
        """
        if not frames:
            return []
        
        results = []
        for i, frame in enumerate(frames):
            frame_id = None
            if frame_ids is not None and i < len(frame_ids):
                frame_id = frame_ids[i]
            
            processed = self.process_frame(frame, frame_id)
            results.append(processed)
        
        return results
    
    def _preprocess(self, frame):
        """
        Apply preprocessing operations to a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Extract configuration
        resize_dim = self.preprocessing_config.get('resize_dim')
        normalize = self.preprocessing_config.get('normalize', True)
        denoise = self.preprocessing_config.get('denoise', True)
        denoise_method = self.preprocessing_config.get('denoise_method', 'gaussian')
        enhance = self.preprocessing_config.get('enhance', True)
        equalize_hist = self.preprocessing_config.get('equalize_hist', False)
        
        # Apply preprocessing using the utility function
        processed = preprocess_frame(
            frame, 
            resize_dim=resize_dim,
            normalize=normalize,
            denoise=denoise,
            enhance=enhance,
            equalize_hist=equalize_hist
        )
        
        return processed
    
    def _update_stats(self, processing_time):
        """
        Update processing statistics.
        
        Args:
            processing_time: Time taken to process the current frame
        """
        self.processing_stats['frames_processed'] += 1
        self.processing_stats['processing_time'] += processing_time
        
        # Calculate running average
        avg_time = (self.processing_stats['processing_time'] / 
                     self.processing_stats['frames_processed'])
        self.processing_stats['avg_processing_time'] = avg_time
        
        # Log statistics periodically
        if self.processing_stats['frames_processed'] % 100 == 0:
            self.logger.info(
                f"Processed {self.processing_stats['frames_processed']} frames. "
                f"Avg processing time: {avg_time*1000:.2f} ms per frame"
            )
    
    def _cache_frame(self, frame_id, processed_frame):
        """
        Cache a processed frame.
        
        Args:
            frame_id: Identifier for the frame
            processed_frame: The processed frame to cache
        """
        # Add to cache
        self.frame_cache[frame_id] = processed_frame
        
        # Maintain maximum cache size
        if len(self.frame_cache) > self.max_cache_size:
            # Remove oldest entry (assuming frame_id is sequential or timestamp)
            oldest_key = min(self.frame_cache.keys())
            del self.frame_cache[oldest_key]
    
    def get_cached_frame(self, frame_id):
        """
        Retrieve a frame from the cache.
        
        Args:
            frame_id: Identifier for the frame
            
        Returns:
            Cached frame or None if not found
        """
        return self.frame_cache.get(frame_id)
    
    def clear_cache(self):
        """Clear the frame cache."""
        self.frame_cache.clear()
        self.logger.info("Frame cache cleared")
    
    def get_processing_stats(self):
        """
        Get current processing statistics.
        
        Returns:
            Dictionary of processing statistics
        """
        return self.processing_stats
    
    def update_config(self, new_config):
        """
        Update preprocessing configuration.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        # Update configuration
        self.preprocessing_config.update(new_config)
        self.logger.info("Preprocessing configuration updated:")
        for key, value in new_config.items():
            self.logger.info(f"  {key}: {value}")


# Integration with VideoCaptureManager example
if __name__ == "__main__":
    from models.video_capture import VideoCaptureManager
    
    # Create video capture manager
    capture_manager = VideoCaptureManager()
    
    # Create frame processor with custom configuration
    processor = FrameProcessor({
        'resize_dim': (640, 480),
        'normalize': False,  # Keep uint8 for display
        'denoise': True,
        'denoise_method': 'gaussian',
        'enhance': True,
        'equalize_hist': False
    })
    
    # Open video source
    if capture_manager.open_source(0):  # Use default camera
        print("Processing video frames. Press 'q' to exit.")
        
        # Process and display frames
        for _, (success, frame, frame_num) in enumerate(capture_manager.read_frames()):
            if not success:
                break
                
            # Process the frame
            processed_frame = processor.process_frame(frame, frame_id=frame_num)
            
            # Display original and processed frames side by side
            if processed_frame is not None:
                # If normalized, convert back to uint8 for display
                if processed_frame.dtype == np.float32 or processed_frame.dtype == np.float64:
                    display_frame = (processed_frame * 255).astype(np.uint8)
                else:
                    display_frame = processed_frame
                    
                # Create side-by-side display
                h, w = frame.shape[:2]
                display_width = 320  # Width of each frame in the display
                display_height = int(h * display_width / w)
                
                # Resize both frames to the same display size
                original_resized = cv2.resize(frame, (display_width, display_height))
                processed_resized = cv2.resize(display_frame, (display_width, display_height))
                
                # Create side-by-side view
                side_by_side = np.zeros((display_height, display_width*2, 3), dtype=np.uint8)
                side_by_side[:, :display_width] = original_resized
                side_by_side[:, display_width:] = processed_resized
                
                # Add labels
                cv2.putText(
                    side_by_side, "Original", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                cv2.putText(
                    side_by_side, "Processed", (display_width + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                
                # Show the side-by-side view
                cv2.imshow('Frame Processing', side_by_side)
                
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Display final statistics
        stats = processor.get_processing_stats()
        print(f"Processed {stats['frames_processed']} frames")
        print(f"Average processing time: {stats['avg_processing_time']*1000:.2f} ms per frame")
        
        # Cleanup
        capture_manager.close()
        cv2.destroyAllWindows()
    else:
        print("Failed to open video source") 