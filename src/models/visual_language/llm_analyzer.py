# src/models/visual_language/llm_analyzer.py

import os
import torch

class LLMAnalyzer:
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
        elif self.model_type == 'local' and self.model_name == 'llama':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = "meta-llama/Llama-3-8B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=self.device,
                torch_dtype=torch.float16
            )
            return {"model": model, "tokenizer": tokenizer}
            
    def analyze_alerts(self, alerts, context=None):
        """分析告警并提供深度解释"""
        if not alerts:
            return None
            
        alert_descriptions = "\n".join([f"- {a['type']} (置信度: {a['confidence']:.2f})" for a in alerts])
        prompt = f"""
        分析以下视频监控告警:
        {alert_descriptions}
        
        提供:
        1. 告警严重程度评估
        2. 可能的原因分析
        3. 建议采取的行动
        4. 是否可能存在误报
        """
        
        if self.model_type == 'api' and self.model_name == 'openai':
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "你是视频监控系统的AI分析师，专注于解释和分析视频监控告警。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
            
        elif self.model_type == 'local' and self.model_name == 'llama':
            tokenizer = self.client["tokenizer"]
            model = self.client["model"]
            
            messages = [
                {"role": "system", "content": "你是视频监控系统的AI分析师，专注于解释和分析视频监控告警。"},
                {"role": "user", "content": prompt}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.9,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的回复（去掉提示部分）
            return response.split("[/INST]")[-1].strip()