#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for motion feature extraction
"""

import os
import sys
import logging
import cv2
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_motion_features():
    """Test motion feature extraction"""
    logger.info("Testing motion feature extraction...")
    
    try:
        from models.motion.motion_manager import MotionFeatureManager
        from models.motion.optical_flow import OpticalFlowExtractor
        from models.motion.motion_history import MotionHistoryExtractor
        
        # Initialize motion feature manager
        motion_manager = MotionFeatureManager(
            use_optical_flow=True,
            use_motion_history=True,
            use_gpu=False
        )
        
        logger.info("Motion feature manager initialized successfully.")
        logger.info(f"Is initialized: {motion_manager.is_initialized}")
        
        # Create test frames (simple moving square)
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a square in frame1
        cv2.rectangle(frame1, (100, 100), (150, 150), (255, 255, 255), -1)
        
        # Draw the same square in a different position in frame2
        cv2.rectangle(frame2, (120, 120), (170, 170), (255, 255, 255), -1)
        
        # Extract features
        logger.info("Extracting motion features...")
        features = motion_manager.extract_features(frame2, frame1)
        
        logger.info(f"Extracted {len(features)} motion features.")
        
        # Visualize features
        logger.info("Visualizing motion features...")
        vis_frame = motion_manager.visualize_features(frame2, features)
        
        # Save visualization frame
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "motion_features.jpg")
        cv2.imwrite(output_path, vis_frame)
        
        logger.info(f"Saved visualization to {output_path}")
        
        # Test with webcam if available
        logger.info("Testing with webcam if available...")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("Webcam not available. Skipping webcam test.")
                return
            
            logger.info("Webcam opened successfully. Press 'q' to quit.")
            
            ret, prev_frame = cap.read()
            if not ret:
                logger.warning("Failed to read from webcam. Skipping webcam test.")
                cap.release()
                return
            
            for i in range(100):  # Process 100 frames max
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                # Extract and visualize features
                features = motion_manager.extract_features(curr_frame, prev_frame)
                vis_frame = motion_manager.visualize_features(curr_frame, features)
                
                # Display frame
                cv2.imshow("Motion Features", vis_frame)
                
                # Save a frame every 30 frames
                if i % 30 == 0:
                    output_path = os.path.join(output_dir, f"webcam_motion_{i}.jpg")
                    cv2.imwrite(output_path, vis_frame)
                    logger.info(f"Saved frame to {output_path}")
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                prev_frame = curr_frame.copy()
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error in webcam test: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error testing motion features: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main entry point"""
    logger.info("Starting motion feature test script...")
    
    test_motion_features()
    
    logger.info("Motion feature test completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 