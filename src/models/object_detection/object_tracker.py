import cv2
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union, Any


class Detection:
    """Class representing a detected object."""
    
    def __init__(self, x1, y1, x2, y2, class_id=0, class_name="unknown", confidence=0.0, id=None):
        """
        Initialize a Detection object.
        
        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            class_id: Class ID
            class_name: Class name
            confidence: Detection confidence
            id: Object ID (optional, used for tracking)
        """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.id = id
    
    @classmethod
    def from_dict(cls, detection_dict):
        """
        Create a Detection object from a detection dictionary.
        
        Args:
            detection_dict: Dictionary with detection data
            
        Returns:
            Detection object
        """
        x1, y1, x2, y2 = detection_dict['box']
        return cls(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            class_id=detection_dict.get('class_id', 0),
            class_name=detection_dict.get('class_name', 'unknown'),
            confidence=detection_dict.get('score', 0.0),
            id=detection_dict.get('id')
        )
    
    def to_dict(self):
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'box': [self.x1, self.y1, self.x2, self.y2],
            'class_id': self.class_id,
            'class_name': self.class_name,
            'score': self.confidence,
            'id': self.id
        }
    
    def get_center(self):
        """
        Get the center point of the detection.
        
        Returns:
            (center_x, center_y)
        """
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def get_area(self):
        """
        Get the area of the detection.
        
        Returns:
            Area in pixels
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def __repr__(self):
        return f"Detection(id={self.id}, class={self.class_name}, conf={self.confidence:.2f})"


class DetectionModel:
    """Base class for object detection models."""
    
    def __init__(self):
        self.is_initialized = False
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries with keys:
            - 'box': [x1, y1, x2, y2]
            - 'score': confidence score
            - 'class_id': class ID
            - 'class_name': class name
        """
        raise NotImplementedError("Subclasses must implement this method")


class ObjectTracker:
    """Base class for object tracking."""
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4, use_gpu=False):
        """
        Initialize the object tracker.
        
        Args:
            confidence_threshold: Confidence threshold for detection
            nms_threshold: Non-maximum suppression threshold
            use_gpu: Whether to use GPU for detection
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ObjectTracker')
        
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        
        # Initialize detector and tracker components
        self.detector = OpenCVDetector(
            model_type='yolov4',
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            device='cuda' if use_gpu else 'cpu'
        )
        
        self.centroid_tracker = CentroidTracker(
            max_disappeared=30,
            max_distance=50.0
        )
        
        self.is_initialized = self.detector.is_initialized
        self.frame_count = 0
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of Detection objects
        """
        if not self.is_initialized:
            self.logger.error("Detector not initialized")
            return []
        
        # Run detector
        detection_dicts = self.detector.detect(frame)
        
        # Convert to Detection objects
        detections = []
        for det_dict in detection_dicts:
            detection = Detection.from_dict(det_dict)
            detections.append(detection)
            
        return detections
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of tracked Detection objects
        """
        if not self.is_initialized:
            self.logger.error("Tracker not initialized")
            return []
        
        # Convert Detection objects to dictionaries for the tracker
        detection_dicts = [det.to_dict() for det in detections]
        
        # Update tracker
        track_dicts = self.centroid_tracker.update(detection_dicts)
        
        # Convert track dictionaries to Detection objects
        tracked_detections = []
        for track_dict in track_dicts:
            detection = Detection.from_dict(track_dict)
            tracked_detections.append(detection)
            
        return tracked_detections
    
    def reset(self):
        """Reset the tracker state."""
        self.centroid_tracker.reset()


class OpenCVDetector(DetectionModel):
    """Object detector using OpenCV's DNN module with pre-trained models."""
    
    SUPPORTED_MODELS = ['yolov4', 'ssd_mobilenet', 'faster_rcnn']
    
    def __init__(self, model_type: str = 'yolov4', confidence_threshold: float = 0.5, 
                 nms_threshold: float = 0.4, device: str = None):
        """
        Initialize the OpenCV-based detector.
        
        Args:
            model_type: Type of model to use (yolov4, ssd_mobilenet, faster_rcnn)
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        super().__init__()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('OpenCVDetector')
        
        self.model_type = model_type.lower()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model based on type
        if self.model_type not in self.SUPPORTED_MODELS:
            self.logger.error(f"Unsupported model type: {model_type}")
            return
        
        try:
            self._load_model()
            self.is_initialized = True
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
    
    def _load_model(self):
        """Load the detection model."""
        # This implementation uses models available in OpenCV's DNN module
        if self.model_type == 'yolov4':
            # YOLOv4 model
            self.net = cv2.dnn.readNet('models/weights/yolov4.weights', 'models/weights/yolov4.cfg')
            
            # Load class names
            with open('models/weights/coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Output layer names
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Set backend and target
            if self.device == 'cuda':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
        elif self.model_type == 'ssd_mobilenet':
            # SSD MobileNet model
            self.net = cv2.dnn.readNetFromTensorflow(
                'models/weights/ssd_mobilenet_v2_coco.pb',
                'models/weights/ssd_mobilenet_v2_coco.pbtxt'
            )
            
            # Load class names
            with open('models/weights/coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Set backend and target
            if self.device == 'cuda':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
        elif self.model_type == 'faster_rcnn':
            # Faster R-CNN model
            self.net = cv2.dnn.readNetFromTensorflow(
                'models/weights/faster_rcnn_inception_v2_coco.pb',
                'models/weights/faster_rcnn_inception_v2_coco.pbtxt'
            )
            
            # Load class names
            with open('models/weights/coco.names', 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Set backend and target
            if self.device == 'cuda':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.logger.info(f"Loaded {self.model_type} model with {len(self.classes)} classes")
    
    def _check_model_files_exist(self) -> bool:
        """Check if model files exist and can be loaded."""
        # This would be implemented to verify model files exist before loading
        # For simplicity, we assume files exist in this example
        return True
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detection dictionaries
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return []
        
        height, width = frame.shape[:2]
        detections = []
        
        try:
            if self.model_type == 'yolov4':
                # Create blob from image
                blob = cv2.dnn.blobFromImage(
                    frame, 1/255.0, (416, 416), swapRB=True, crop=False
                )
                self.net.setInput(blob)
                
                # Run forward pass
                outputs = self.net.forward(self.output_layers)
                
                # Process detections
                boxes = []
                confidences = []
                class_ids = []
                
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        
                        if confidence > self.confidence_threshold:
                            # Scale bounding box coordinates to image size
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            # Get top-left coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                # Apply non-maximum suppression
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                
                # Create detection dictionaries
                for i in indices:
                    if isinstance(i, tuple) or isinstance(i, list):
                        i = i[0]  # Handle different OpenCV versions
                    
                    box = boxes[i]
                    x, y, w, h = box
                    
                    detections.append({
                        'box': [x, y, x + w, y + h],
                        'score': confidences[i],
                        'class_id': class_ids[i],
                        'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else f"class_{class_ids[i]}"
                    })
                    
            elif self.model_type == 'ssd_mobilenet' or self.model_type == 'faster_rcnn':
                # Create blob from image
                blob = cv2.dnn.blobFromImage(
                    frame, size=(300, 300), swapRB=True, crop=False
                )
                self.net.setInput(blob)
                
                # Run forward pass
                output = self.net.forward()
                
                # Process detections
                for i in range(output.shape[2]):
                    confidence = output[0, 0, i, 2]
                    
                    if confidence > self.confidence_threshold:
                        class_id = int(output[0, 0, i, 1])
                        
                        # Scale bounding box coordinates to image size
                        box = output[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'score': float(confidence),
                            'class_id': class_id,
                            'class_name': self.classes[class_id - 1] if 0 <= class_id - 1 < len(self.classes) else f"class_{class_id}"
                        })
        
        except Exception as e:
            self.logger.error(f"Error during detection: {str(e)}")
        
        return detections


class CentroidTracker(ObjectTracker):
    """
    Centroid-based object tracker.
    """
    
    def __init__(self, max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the centroid tracker.
        
        Args:
            max_disappeared: Maximum number of frames an object can be missing before deregistering
            max_distance: Maximum distance between centroids to consider it the same object
        """
        # 不要调用super().__init__()来避免递归，而是直接初始化所需变量
        # 配置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CentroidTracker')
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Initialize ID counter
        self.next_object_id = 0
        
        # Track dictionaries
        self.objects = {}  # ID -> centroid
        self.disappeared = {}  # ID -> number of frames disappeared
        self.object_data = {}  # ID -> detection data
        
        self.is_initialized = True
    
    def _get_centroid(self, box: List[int]) -> Tuple[int, int]:
        """
        Calculate centroid from box coordinates.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2]
            
        Returns:
            Centroid coordinates (x, y)
        """
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update(self, detections: List[Dict[str, Any]], frame: np.ndarray = None) -> List[Dict[str, Any]]:
        """
        Update object tracks with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (not used in centroid tracker)
            
        Returns:
            List of track dictionaries
        """
        # If no detections, increment disappeared count for all objects
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                # Deregister if object has disappeared for too long
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            
            # Return current tracks
            return self._get_tracks()
        
        # Initialize centroids for current detections
        centroids = []
        for detection in detections:
            centroid = self._get_centroid(detection['box'])
            centroids.append(centroid)
        
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for i, (centroid, detection) in enumerate(zip(centroids, detections)):
                self._register(centroid, detection)
        
        # Otherwise, match detections to existing objects
        else:
            # Get existing object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distances between existing objects and new detections
            distances = np.zeros((len(object_centroids), len(centroids)))
            for i, object_centroid in enumerate(object_centroids):
                for j, centroid in enumerate(centroids):
                    distances[i, j] = np.sqrt(
                        (object_centroid[0] - centroid[0]) ** 2 +
                        (object_centroid[1] - centroid[1]) ** 2
                    )
            
            # Find the minimum distance for each row (existing object)
            # and sort by minimum distance
            rows = distances.min(axis=1).argsort()
            
            # Find the minimum distance for each column (detection)
            cols = distances.argmin(axis=0)
            
            # Keep track of which rows and columns have been matched
            used_rows = set()
            used_cols = set()
            
            # Match objects to detections
            for row in rows:
                # Get the column with the minimum distance for this row
                col = distances[row].argmin()
                
                # If the column has been used or the distance is too large, skip
                if col in used_cols or distances[row, col] > self.max_distance:
                    continue
                
                # Get the object ID for this row
                object_id = object_ids[row]
                
                # Update the object with the new centroid and detection data
                self.objects[object_id] = centroids[col]
                self.object_data[object_id] = detections[col]
                self.disappeared[object_id] = 0
                
                # Mark row and column as used
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle unused rows (disappeared objects)
            for row in range(len(object_ids)):
                if row not in used_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    # Deregister if object has disappeared for too long
                    if self.disappeared[object_id] > self.max_disappeared:
                        self._deregister(object_id)
            
            # Handle unused columns (new detections)
            for col in range(len(centroids)):
                if col not in used_cols:
                    self._register(centroids[col], detections[col])
        
        # Return current tracks
        return self._get_tracks()
    
    def _register(self, centroid: Tuple[int, int], detection: Dict[str, Any]):
        """
        Register a new object.
        
        Args:
            centroid: Centroid coordinates (x, y)
            detection: Detection dictionary
        """
        # Assign new ID and store object data
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.object_data[self.next_object_id] = detection
        
        # Increment ID counter
        self.next_object_id += 1
    
    def _deregister(self, object_id: int):
        """
        Deregister an object.
        
        Args:
            object_id: Object ID to deregister
        """
        # Remove object from dictionaries
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.object_data[object_id]
    
    def _get_tracks(self) -> List[Dict[str, Any]]:
        """
        Get current tracks.
        
        Returns:
            List of track dictionaries
        """
        tracks = []
        
        for object_id in self.objects.keys():
            # Get detection data
            detection = self.object_data[object_id]
            
            # Create track dictionary
            track = {
                'id': object_id,
                'box': detection['box'],
                'score': detection['score'],
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'age': self.max_disappeared - self.disappeared[object_id],
                'centroid': self.objects[object_id]
            }
            
            tracks.append(track)
        
        return tracks
    
    def reset(self):
        """Reset the tracker state."""
        self.objects = {}
        self.disappeared = {}
        self.object_data = {}
        self.next_object_id = 0


class ObjectDetectionAndTracking:
    """
    Integrated object detection and tracking module.
    Combines a detector and tracker to provide unified functionality.
    """
    
    def __init__(self, detector_type: str = 'yolov4', tracker_type: str = 'centroid', 
                 confidence_threshold: float = 0.5, device: str = None,
                 max_disappeared: int = 30, max_distance: float = 50.0):
        """
        Initialize the object detection and tracking module.
        
        Args:
            detector_type: Type of detector to use ('yolov4', 'ssd_mobilenet', 'faster_rcnn')
            tracker_type: Type of tracker to use ('centroid')
            confidence_threshold: Minimum confidence for detection
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
            max_disappeared: Maximum number of frames an object can be missing before deregistering
            max_distance: Maximum distance for tracker
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ObjectDetectionAndTracking')
        
        # Initialize detector
        if detector_type == 'yolov4':
            self.detector = OpenCVDetector(
                model_type='yolov4',
                confidence_threshold=confidence_threshold,
                device=device
            )
        elif detector_type == 'ssd_mobilenet':
            self.detector = OpenCVDetector(
                model_type='ssd_mobilenet',
                confidence_threshold=confidence_threshold,
                device=device
            )
        elif detector_type == 'faster_rcnn':
            self.detector = OpenCVDetector(
                model_type='faster_rcnn',
                confidence_threshold=confidence_threshold,
                device=device
            )
        else:
            self.logger.error(f"Unsupported detector type: {detector_type}")
            return
        
        # Initialize tracker
        if tracker_type == 'centroid':
            self.tracker = CentroidTracker(
                max_disappeared=max_disappeared,
                max_distance=max_distance
            )
        else:
            self.logger.error(f"Unsupported tracker type: {tracker_type}")
            return
        
        self.is_initialized = self.detector.is_initialized and self.tracker.is_initialized
        
        # Performance metrics
        self.frame_count = 0
        self.detection_time = 0
        self.tracking_time = 0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, float]:
        """
        Process a single frame for detection and tracking.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, tracks, detection_time, tracking_time)
        """
        if not self.is_initialized:
            self.logger.error("Module not initialized")
            return [], [], 0, 0
        
        self.frame_count += 1
        
        # Run object detection
        start_time = cv2.getTickCount()
        detections = self.detector.detect(frame)
        detection_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self.detection_time += detection_time
        
        # Run object tracking
        start_time = cv2.getTickCount()
        tracks = self.tracker.update(detections, frame)
        tracking_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        self.tracking_time += tracking_time
        
        return detections, tracks, detection_time, tracking_time
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict[str, Any]] = None, 
                    tracks: List[Dict[str, Any]] = None) -> np.ndarray:
        """
        Draw detection and tracking results on a frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries (optional)
            tracks: List of track dictionaries (optional)
            
        Returns:
            Frame with results drawn
        """
        output_frame = frame.copy()
        
        # Draw detections
        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection['box']
                class_name = detection['class_name']
                score = detection['score']
                
                # Draw box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{class_name}: {score:.2f}"
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(output_frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw tracks
        if tracks:
            for track in tracks:
                x1, y1, x2, y2 = track['box']
                track_id = track['id']
                class_name = track['class_name']
                
                # Draw box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw ID and class
                label = f"ID: {track_id} ({class_name})"
                y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                cv2.putText(output_frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw centroid
                if 'centroid' in track:
                    cx, cy = track['centroid']
                    cv2.circle(output_frame, (cx, cy), 4, (0, 0, 255), -1)
        
        # Add performance metrics
        if self.frame_count > 0:
            avg_detection_time = self.detection_time / self.frame_count
            avg_tracking_time = self.tracking_time / self.frame_count
            
            cv2.putText(
                output_frame,
                f"Avg Detection: {avg_detection_time:.4f}s | Avg Tracking: {avg_tracking_time:.4f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        
        return output_frame
    
    def reset(self):
        """Reset the module state."""
        self.tracker.reset()
        self.frame_count = 0
        self.detection_time = 0
        self.tracking_time = 0
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.frame_count == 0:
            return {
                'avg_detection_time': 0,
                'avg_tracking_time': 0,
                'avg_total_time': 0,
                'frame_count': 0
            }
        
        avg_detection_time = self.detection_time / self.frame_count
        avg_tracking_time = self.tracking_time / self.frame_count
        avg_total_time = (self.detection_time + self.tracking_time) / self.frame_count
        
        return {
            'avg_detection_time': avg_detection_time,
            'avg_tracking_time': avg_tracking_time,
            'avg_total_time': avg_total_time,
            'frame_count': self.frame_count
        }