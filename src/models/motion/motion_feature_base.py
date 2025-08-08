import cv2
import numpy as np
import logging
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class MotionFeature:
    """Data class for motion feature representation."""
    type: str                # 特征类型
    data: np.ndarray         # 特征数据
    position: Tuple[int, int] # 特征位置 (x, y)
    frame_idx: int           # 帧索引
    object_id: Optional[int] = None  # 关联的目标ID (如果可用)
    confidence: float = 1.0  # 特征置信度


class MotionFeatureExtractor:
    """Base class for motion feature extractors."""
    
    def __init__(self):
        """Initialize the base motion feature extractor."""
        self.is_initialized = False
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing {self.__class__.__name__}")
        
        # Check OpenCV version since some methods may be dependent on version
        opencv_version = cv2.__version__
        self.logger.info(f"OpenCV version: {opencv_version}")
        
        # Subclasses should set this to True after successful initialization
        # self.is_initialized will be True only if initialization succeeds
    
    def extract(self, frame: np.ndarray, prev_frame: np.ndarray = None, **kwargs) -> List[MotionFeature]:
        """
        Extract motion features from consecutive frames.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of motion features
        """
        if not self.is_initialized:
            self.logger.error(f"{self.__class__.__name__} not properly initialized")
            return []
            
        try:
            # To be implemented by subclasses
            return self._extract_impl(frame, prev_frame, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}.extract: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []
    
    def _extract_impl(self, frame: np.ndarray, prev_frame: np.ndarray = None, **kwargs) -> List[MotionFeature]:
        """
        Implementation of feature extraction to be overridden by subclasses.
        
        Args:
            frame: Current frame
            prev_frame: Previous frame (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of motion features
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement _extract_impl method") 
    
    def reset(self):
        """Reset the extractor state."""
        # To be implemented by subclasses if needed
        pass 