# src/models/visual_language/vision_model.py

import torch
import numpy as np
import cv2

class VisionModelManager:
    def __init__(self, config):
        self.model_name = config.get('model_name', 'yolov8n')
        self.confidence = config.get('confidence', 0.5)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
    def _load_model(self):
        if self.model_name.startswith('yolo'):
            from ultralytics import YOLO
            return YOLO(f"{self.model_name}.pt")
        elif self.model_name == 'clip':
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.to(self.device)
            return {"model": model, "processor": processor}
        # 添加其他模型支持...
            
    def process_frame(self, frame):
        """处理帧并返回检测结果"""
        if self.model_name.startswith('yolo'):
            results = self.model(frame)
            return self._parse_yolo_results(results)
        # 其他模型处理...
    
    def _parse_yolo_results(self, results):
        """将YOLO结果转换为标准格式"""
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf.item() > self.confidence:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': r.names[cls],
                        'confidence': conf
                    })
        return detections