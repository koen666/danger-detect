import cv2
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any

from models.motion.motion_feature_base import MotionFeature, MotionFeatureExtractor


class OpticalFlowExtractor(MotionFeatureExtractor):
    """Extract optical flow features from consecutive frames."""
    
    METHODS = {
        'farneback': cv2.calcOpticalFlowFarneback,
        'pyr_lk': None,  # Implemented separately since it uses a different approach
    }
    
    def __init__(self, method: str = 'farneback', pyr_scale: float = 0.5, 
                 levels: int = 3, winsize: int = 15, iterations: int = 3,
                 poly_n: int = 5, poly_sigma: float = 1.2,
                 use_gpu: bool = False):
        """
        Initialize the optical flow extractor.
        
        Args:
            method: Optical flow method ('farneback' or 'pyr_lk')
            pyr_scale: Image scale for pyramid (<1 means larger pyramids)
            levels: Number of pyramid levels
            winsize: Averaging window size
            iterations: Number of iterations at each pyramid level
            poly_n: Size of pixel neighborhood for polynomial approximation
            poly_sigma: Standard deviation of Gaussian for polynomial approximation
            use_gpu: Use GPU acceleration if available
        """
        super().__init__()
        
        try:
            self.method = method.lower()
            if self.method not in self.METHODS:
                self.logger.error(f"Unsupported optical flow method: {method}")
                return
            
            # Set parameters
            self.pyr_scale = pyr_scale
            self.levels = levels
            self.winsize = winsize
            self.iterations = iterations
            self.poly_n = poly_n
            self.poly_sigma = poly_sigma
            
            # Set device for GPU acceleration
            self.use_gpu = use_gpu
            cuda_available = False
            
            try:
                cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
                if cuda_available:
                    self.logger.info(f"CUDA device count: {cv2.cuda.getCudaEnabledDeviceCount()}")
            except Exception as e:
                self.logger.warning(f"Could not check CUDA availability: {str(e)}")
                cuda_available = False
                
            self.use_gpu = use_gpu and cuda_available
                
            if self.use_gpu and self.method == 'farneback':
                try:
                    self.flow_calculator = cv2.cuda.FarnebackOpticalFlow.create(
                        numLevels=self.levels,
                        pyrScale=self.pyr_scale,
                        winSize=self.winsize,
                        numIters=self.iterations,
                        polyN=self.poly_n,
                        polySigma=self.poly_sigma
                    )
                    self.logger.info("Successfully created GPU flow calculator")
                except Exception as e:
                    self.logger.error(f"Failed to create GPU flow calculator: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    self.use_gpu = False
                    
                if self.use_gpu:
                    self.logger.info("Using GPU acceleration for optical flow")
            elif self.use_gpu:
                self.logger.warning(f"GPU acceleration not available for {self.method} method")
                self.use_gpu = False
            
            # For Lucas-Kanade method
            if self.method == 'pyr_lk':
                # Parameters for ShiTomasi corner detection
                self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
                
                # Parameters for Lucas-Kanade optical flow
                self.lk_params = dict(
                    winSize=(self.winsize, self.winsize),
                    maxLevel=self.levels,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                )
                
                # Create points to track
                self.prev_pts = None
            
            self.is_initialized = True
            self.prev_gray = None
            self.flow = None  # 初始化光流
            self.logger.info(f"OpticalFlowExtractor initialized successfully with method: {self.method}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpticalFlowExtractor: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_initialized = False
    
    def _extract_impl(self, frame: np.ndarray, prev_frame: np.ndarray = None, 
                tracks: List[Dict[str, Any]] = None) -> List[MotionFeature]:
        """
        Extract optical flow features from consecutive frames.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (if None, will use cached previous frame)
            tracks: Object tracks from object tracker (optional)
            
        Returns:
            List of optical flow features
        """
        # Convert frames to grayscale
        if frame is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None and self.prev_gray is None:
            # First frame, no optical flow yet
            self.prev_gray = gray
            
            # For Lucas-Kanade method, initialize points to track
            if self.method == 'pyr_lk':
                self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            
            return []
        
        if prev_frame is not None:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = self.prev_gray
        
        features = []
        
        try:
            if self.method == 'farneback':
                features = self._extract_farneback(prev_gray, gray, tracks)
            elif self.method == 'pyr_lk':
                features = self._extract_lucas_kanade(prev_gray, gray)
        
        except Exception as e:
            self.logger.error(f"Error calculating optical flow: {str(e)}")
            self.logger.error(traceback.format_exc())
        
        # Update previous gray frame
        self.prev_gray = gray
        
        return features
    
    def _extract_farneback(self, prev_gray: np.ndarray, gray: np.ndarray, 
                          tracks: List[Dict[str, Any]] = None) -> List[MotionFeature]:
        """Extract optical flow using Farneback method."""
        features = []
        
        if self.use_gpu:
            # GPU implementation
            prev_cuda = cv2.cuda_GpuMat(prev_gray)
            curr_cuda = cv2.cuda_GpuMat(gray)
            flow_cuda = self.flow_calculator.calc(prev_cuda, curr_cuda, None)
            flow = flow_cuda.download()
        else:
            # CPU implementation
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, self.pyr_scale, self.levels, self.winsize,
                self.iterations, self.poly_n, self.poly_sigma, 0
            )
        
        # 保存计算的光流，以便后续可以访问
        self.flow = flow
        
        # Extract flow features for the entire frame
        step = 16  # Sample flow every 16 pixels
        h, w = flow.shape[:2]
        
        # Sample flow vectors
        y_indices, x_indices = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # Get flow vectors at sampled locations
        fx, fy = flow[y_indices, x_indices].T
        
        # Filter out small movements
        mag = np.sqrt(fx*fx + fy*fy)
        mask = mag > 1.0
        
        # Create motion features
        for i in range(len(x_indices)):
            if mask[i]:
                features.append(MotionFeature(
                    type='optical_flow',
                    data=np.array([fx[i], fy[i]]),
                    position=(int(x_indices[i]), int(y_indices[i])),
                    frame_idx=0,  # Will be set by caller
                    confidence=float(mag[i] / np.max(mag) if np.max(mag) > 0 else 0)
                ))
        
        # If tracks are provided, calculate flow for each track
        if tracks:
            for track in tracks:
                box = track['box']
                track_id = track['id']
                
                # Calculate center of bounding box
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                
                # Get flow at center
                if 0 <= cy < h and 0 <= cx < w:
                    fx, fy = flow[cy, cx]
                    
                    # Only add if there's significant motion
                    mag = np.sqrt(fx*fx + fy*fy)
                    if mag > 1.0:
                        features.append(MotionFeature(
                            type='object_flow',
                            data=np.array([fx, fy]),
                            position=(cx, cy),
                            frame_idx=0,  # Will be set by caller
                            object_id=track_id,
                            confidence=1.0
                        ))
        
        return features
    
    def _extract_lucas_kanade(self, prev_gray: np.ndarray, gray: np.ndarray) -> List[MotionFeature]:
        """Extract optical flow using Lucas-Kanade method."""
        features = []
        
        if self.prev_pts is None or len(self.prev_pts) == 0:
            # Initialize points to track
            self.prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
            return []
        
        # Calculate optical flow using Lucas-Kanade method
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        # Select good points
        if curr_pts is not None:
            good_new = curr_pts[status == 1]
            good_old = self.prev_pts[status == 1]
            
            # Create motion features for each point
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                nx, ny = new.ravel()
                ox, oy = old.ravel()
                
                # Calculate displacement
                dx = nx - ox
                dy = ny - oy
                
                # Calculate magnitude
                mag = np.sqrt(dx*dx + dy*dy)
                
                # Only add if there's significant motion
                if mag > 1.0:
                    features.append(MotionFeature(
                        type='sparse_flow',
                        data=np.array([dx, dy]),
                        position=(int(nx), int(ny)),
                        frame_idx=0,  # Will be set by caller
                        confidence=float(1.0 / (err[i][0] + 1e-6))  # Use inverse of error as confidence
                    ))

            
            # Update previous points
            self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            # Reset points if tracking fails
            self.prev_pts = None
        
        return features
    
    def reset(self):
        """Reset the optical flow extractor."""
        self.prev_gray = None
        self.flow = None
        self.logger.info("Optical flow extractor reset")
        
    def get_flow(self):
        """
        获取当前计算的光流
        
        Returns:
            np.ndarray: 光流场或None（如果尚未计算）
        """
        return self.flow 