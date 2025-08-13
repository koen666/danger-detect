# src/models/visual_language/multimodal_manager.py

import os
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

class MultimodalManager:
    def __init__(self, config):
        self.model_type = config.get('type', 'api')
        self.model_name = config.get('name', 'openai')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.client = self._setup_client()
        
    def _setup_client(self):
        if self.model_type == 'api' and self.model_name == 'openai':
            import openai
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            return openai.Client()
        elif self.model_type == 'api' and self.model_name == 'anthropic':
            import anthropic
            return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        elif self.model_type == 'local' and self.model_name == 'qwen-vl':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map=self.device, trust_remote_code=True)
            return {"model": model, "tokenizer": tokenizer}
            
    def analyze_scene(self, frame, detections=None):
        """使用多模态模型分析场景"""
        if self.model_type == 'api' and self.model_name == 'openai':
            # 将帧转换为base64
            buffered = BytesIO()
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            detection_text = ""
            if detections:
                detection_text = "检测到的物体:\n" + "\n".join([
                    f"- {d['class']} (位置: [{d['bbox'][0]},{d['bbox'][1]}]-[{d['bbox'][2]},{d['bbox'][3]}], 置信度: {d['confidence']:.2f})"
                    for d in detections
                ])
            
            # 调用API
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {"role": "system", "content": "分析视频画面中的潜在危险行为和异常情况。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"分析这个视频帧中是否存在危险行为或异常情况:\n{detection_text}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        
        elif self.model_type == 'local' and self.model_name == 'qwen-vl':
            # 处理本地Qwen-VL模型
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            query = "分析这个视频帧中是否存在危险行为或异常情况。"
            
            tokenizer = self.client["tokenizer"]
            model = self.client["model"]
            
            query_with_image = tokenizer.from_list_format([
                {'image': image},
                {'text': query},
            ])
            response, _ = model.chat(tokenizer, query=query_with_image, history=None)
            return response