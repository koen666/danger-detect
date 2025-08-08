import cv2
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Any
import time

from models.motion.motion_feature_base import MotionFeature, MotionFeatureExtractor


class MotionHistoryExtractor(MotionFeatureExtractor):
    """Extract motion history features from consecutive frames."""
    
    def __init__(self, history_length: int = 20, threshold: int = 30,
                 max_val: float = 1.0, min_val: float = 0.05):
        """
        Initialize the motion history extractor.
        
        Args:
            history_length: Length of motion history in frames
            threshold: Threshold for motion detection
            max_val: Maximum value of motion history
            min_val: Minimum value of motion history
        """
        super().__init__()
        
        try:
            self.logger.info("Initializing MotionHistoryExtractor...")
            self.history_length = history_length
            self.threshold = threshold
            self.max_val = max_val
            self.min_val = min_val
            
            # Initialize motion history image
            self.mhi = None
            self.prev_gray = None
            self.last_timestamp = 0
            
            # Check if motempl module is available
            self.use_legacy_motempl = False
            try:
                if hasattr(cv2, 'motempl'):
                    self.use_legacy_motempl = True
                    self.logger.info("Using legacy cv2.motempl module")
                else:
                    self.logger.info("cv2.motempl module not available, using custom implementation")
            except Exception as e:
                self.logger.info(f"cv2.motempl module check failed: {str(e)}, using custom implementation")
            
            self.is_initialized = True
            self.logger.info("MotionHistoryExtractor initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MotionHistoryExtractor: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.is_initialized = False
    
    def update_motion_history(self, motion_mask, mhi, timestamp, duration):
        """
        自定义实现的updateMotionHistory功能。
        
        Args:
            motion_mask: 运动掩码（二值图像）
            mhi: 运动历史图像（如果不存在则初始化）
            timestamp: 当前时间戳
            duration: 历史保留时长
            
        Returns:
            更新后的运动历史图像
        """
        # 如果运动历史为空，则初始化
        if mhi is None:
            h, w = motion_mask.shape
            mhi = np.zeros((h, w), dtype=np.float32)
            self.logger.info(f"Initialized motion history image: {w}x{h}")
        
        # 创建副本以防止修改原图像
        mhi_copy = mhi.copy()
        
        # 更新MHI：设置有运动的区域为当前时间戳
        mhi_copy[motion_mask > 0] = timestamp
        
        # 删除超过持续时间的部分
        old_vals = (timestamp - mhi_copy) > duration
        mhi_copy[old_vals] = 0
        
        return mhi_copy
    
    def _extract_impl(self, frame, prev_frame, **kwargs):
        """
        Implementation of the motion history extraction.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame
            **kwargs: Additional parameters including tracks
            
        Returns:
            List of motion features
        """
        tracks = kwargs.get('tracks', None)
        
        if frame is None or prev_frame is None:
            return []
        
        # Convert frames to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
            
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame.copy()
            
        # 确保两个图像有相同的数据类型
        if gray.dtype != prev_gray.dtype:
            self.logger.debug(f"Converting prev_gray from {prev_gray.dtype} to {gray.dtype}")
            prev_gray = prev_gray.astype(gray.dtype)
            
        # Compute frame difference
        try:
            frame_diff = cv2.absdiff(gray, prev_gray)
        except cv2.error as e:
            self.logger.error(f"absdiff error: {str(e)} - gray: {gray.dtype}, prev_gray: {prev_gray.dtype}")
            # 尝试强制转换为相同类型
            prev_gray = prev_gray.astype(np.uint8)
            gray = gray.astype(np.uint8)
            frame_diff = cv2.absdiff(gray, prev_gray)

        # Threshold the difference
        _, motion_mask = cv2.threshold(frame_diff, self.threshold, 1, cv2.THRESH_BINARY)
        motion_mask = motion_mask.astype(np.uint8)
        
        # Update timestamp
        timestamp = time.time()
        if self.last_timestamp == 0:
            self.last_timestamp = timestamp
        
        # 初始化mhi如果需要
        h, w = motion_mask.shape
        if self.mhi is None or self.mhi.shape != motion_mask.shape:
            self.mhi = np.zeros((h, w), dtype=np.float32)
            self.logger.info(f"Initialized MHI: {w}x{h}")
        
        # Update motion history
        self.mhi = self.update_motion_history(motion_mask, self.mhi, timestamp, self.history_length)
        
        # Save the previous grayscale image
        self.prev_gray = gray
        
        # Calculate motion features on a grid
        h, w = self.mhi.shape
        step = 16  # Grid step
        y_indices, x_indices = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        features = []
        
        try:
            # Get motion history at sampled locations
            for i, (x, y) in enumerate(zip(x_indices, y_indices)):
                # Skip if outside frame
                if x >= w or y >= h:
                    continue
                
                # Get motion value from MHI
                motion_val = self.mhi[y, x]
                
                # Skip if no motion
                if motion_val <= 0:
                    continue
                
                # Calculate orientation
                # Use Sobel derivatives to get orientation if available
                gradient_size = 3
                try:
                    dx = cv2.Sobel(self.mhi, cv2.CV_32F, 1, 0, ksize=gradient_size)
                    dy = cv2.Sobel(self.mhi, cv2.CV_32F, 0, 1, ksize=gradient_size)
                    
                    mag = np.sqrt(dx[y, x]**2 + dy[y, x]**2)
                    angle = np.arctan2(dy[y, x], dx[y, x]) * 180 / np.pi  # Convert to degrees
                    
                except Exception as e:
                    self.logger.error(f"Error calculating orientation: {str(e)}")
                    mag = 0
                    angle = 0
                
                # Create motion feature
                from models.motion.motion_feature_base import MotionFeature
                

                
                feature = MotionFeature(
                    position=(x, y),
                    type="motion_history",
                    data=[motion_val, mag, angle],
                    frame_idx=None  # Will be set by manager
                )
                
                features.append(feature)
        
        except Exception as e:
            self.logger.error(f"Error extracting grid features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return features
    
    def extract(self, frame, prev_frame=None, tracks=None):
        """
        Extract motion history features from frame.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (if None, will use cached previous frame)
            tracks: Object tracks from object tracker (optional)
            
        Returns:
            List of motion features
        """
        if not self.is_initialized:
            self.logger.error("Extractor not initialized")
            return []
        
        try:
            # 如果prev_frame是None且我们有缓存的前一帧，使用它
            if prev_frame is None:
                prev_frame = self.prev_gray
                
            # 如果是第一帧，或者前一帧不可用，初始化并返回空结果
            if prev_frame is None:
                # 转换当前帧为灰度并缓存
                if len(frame.shape) == 3:
                    self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    self.prev_gray = frame.copy()
                
                # 初始化运动历史图像
                h, w = self.prev_gray.shape
                self.mhi = np.zeros((h, w), dtype=np.float32)
                self.logger.info(f"初始化运动历史图像: {w}x{h}")
                
                return []
                
            return self._extract_impl(frame, prev_frame, tracks=tracks)
        except Exception as e:
            self.logger.error(f"Error extracting motion history features: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def get_motion_history_image(self) -> np.ndarray:
        """
        Get the current motion history image.
        
        Returns:
            Motion history image normalized to [0, 255] for visualization
        """
        if self.mhi is None:
            return None
        
        # Normalize to [0, 255]
        return cv2.normalize(self.mhi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def reset(self):
        """Reset the motion history extractor."""
        self.prev_gray = None
        self.mhi = None
        self.last_timestamp = 0
        self.logger.info("Motion history extractor reset")
        
    def get_motion_history_image(self):
        """
        获取当前的运动历史图像
        
        Returns:
            np.ndarray: 运动历史图像或None（如果尚未初始化）
        """
        return self.mhi