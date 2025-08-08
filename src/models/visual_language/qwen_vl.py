import os
import torch
import numpy as np
import logging
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig


class QwenVLFeatureExtractor:
    """
    A class for extracting visual-language features using the Qwen-VL model.
    This class provides interfaces for image captioning, VQA, and feature extraction.
    """
    
    def __init__(self, model_version="Qwen/Qwen-VL-Chat", device=None):
        """
        Initialize the Qwen-VL feature extractor.
        
        Args:
            model_version: Model version to use
            device: Device to run the model on ("cuda" or "cpu")
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('QwenVLFeatureExtractor')
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.logger.info(f"Loading model: {model_version}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_version, trust_remote_code=True)
            
            # Load model with half precision to save memory
            self.model = AutoModelForCausalLM.from_pretrained(
                model_version,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.float16
            ).eval()
            
            # Set generation config
            self.model.generation_config = GenerationConfig.from_pretrained(model_version, trust_remote_code=True)
            
            self.logger.info("Model loaded successfully")
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.is_initialized = False
    
    def _frame_to_pil(self, frame):
        """
        Convert OpenCV frame to PIL Image.
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            PIL Image (RGB format)
        """
        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1]
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        return pil_image
    
    def _prepare_query_with_image(self, frame, text):
        """
        Prepare a query with an image for the model.
        
        Args:
            frame: OpenCV frame
            text: Text prompt
        
        Returns:
            Tokenized query
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return None
        
        # Convert frame to PIL Image
        pil_image = self._frame_to_pil(frame)
        
        # Prepare query with image
        query = self.tokenizer.from_messages([
            {"role": "user", "content": [{"image": pil_image}, {"text": text}]}
        ], return_tensors="pt")
        
        # Move query to device
        query = {k: v.to(self.device) for k, v in query.items()}
        
        return query
    
    def extract_features(self, frame, text_context=None):
        """
        Extract multimodal features from a frame with optional text context.
        
        Args:
            frame: OpenCV frame
            text_context: Optional text context
        
        Returns:
            Feature vector
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return None
        
        if frame is None:
            self.logger.error("Cannot extract features from None frame")
            return None
        
        # Default text context if none provided
        if text_context is None:
            text_context = "Describe this image in detail."
        
        try:
            # Prepare query with image
            query = self._prepare_query_with_image(frame, text_context)
            
            if query is None:
                return None
                
            # Get model embedding
            with torch.no_grad():
                # Extract the hidden states from the model
                outputs = self.model(**query, output_hidden_states=True)
                
                # Get the last hidden state features
                hidden_states = outputs.hidden_states[-1]
                
                # Average the features across all tokens (excluding padding)
                attention_mask = query['attention_mask']
                features = hidden_states * attention_mask.unsqueeze(-1)
                features = features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Convert to numpy array
                features = features.cpu().numpy()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def generate_caption(self, frame, prompt="Describe this image in detail."):
        """
        Generate a caption for a frame.
        
        Args:
            frame: OpenCV frame
            prompt: Prompt for caption generation
        
        Returns:
            Generated caption
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return "Model not initialized"
        
        if frame is None:
            self.logger.error("Cannot generate caption for None frame")
            return "No frame provided"
        
        try:
            # Prepare query with image
            query = self._prepare_query_with_image(frame, prompt)
            
            if query is None:
                return "Failed to prepare query"
                
            # Generate response
            generation_output = self.model.generate(
                **query,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
            )
            
            # Decode the output
            response = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
            
            # Extract the assistant's response
            response = response.split("ASSISTANT: ", 1)[-1].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {str(e)}")
            return f"Error: {str(e)}"
    
    def answer_question(self, frame, question):
        """
        Answer a question about a frame.
        
        Args:
            frame: OpenCV frame
            question: Question about the frame
        
        Returns:
            Answer to the question
        """
        # This is just a wrapper around generate_caption with the question as prompt
        return self.generate_caption(frame, question)
    
    def classify_scene(self, frame, categories):
        """
        Classify a frame according to provided categories.
        
        Args:
            frame: OpenCV frame
            categories: List of categories to classify into
            
        Returns:
            Dictionary of category scores
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return None
        
        if frame is None:
            self.logger.error("Cannot classify None frame")
            return None
        
        # Join categories into a string
        categories_str = ", ".join(categories)
        
        # Create a prompt for classification
        prompt = f"Classify this image into exactly one of these categories: {categories_str}. Answer with just the category name."
        
        # Get the response
        response = self.generate_caption(frame, prompt)
        
        # Find the best matching category
        best_match = None
        best_score = 0
        
        for category in categories:
            if category.lower() in response.lower():
                # Simple match - more sophisticated matching could be implemented
                score = 1.0
                if best_score < score:
                    best_score = score
                    best_match = category
        
        # If no match found, use the first word as a guess
        if best_match is None:
            words = response.split()
            if words:
                best_match = words[0]
                best_score = 0.5
        
        return {
            "category": best_match,
            "score": best_score,
            "raw_response": response
        }
    
    def detect_anomalies(self, frame, normal_context):
        """
        Detect anomalies in a frame based on a description of normal conditions.
        
        Args:
            frame: OpenCV frame
            normal_context: Description of normal conditions
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return None
        
        if frame is None:
            self.logger.error("Cannot detect anomalies in None frame")
            return None
        
        # Create a prompt for anomaly detection
        prompt = f"""
        In a normal scenario: {normal_context}
        
        Look at this image and determine if there's anything abnormal or unusual compared to the description above.
        If you detect any anomalies, describe them in detail. If everything looks normal, just say "No anomalies detected."
        """
        
        # Get the response
        response = self.generate_caption(frame, prompt)
        
        # Check if anomalies were detected
        has_anomalies = "no anomalies detected" not in response.lower()
        
        return {
            "has_anomalies": has_anomalies,
            "description": response
        }
    
    def get_feature_dimension(self):
        """Get the dimension of the feature vectors."""
        if not self.is_initialized:
            self.logger.error("Model not initialized")
            return 0
        
        # This is the dimension of the Qwen-VL hidden states
        return self.model.config.hidden_size


# Simple test function
if __name__ == "__main__":
    import cv2
    import time
    
    # Initialize feature extractor
    extractor = QwenVLFeatureExtractor()
    
    if not extractor.is_initialized:
        print("Failed to initialize feature extractor")
        exit()
    
    # Load an image or capture from camera
    # Option 1: Load image
    # frame = cv2.imread("path/to/image.jpg")
    
    # Option 2: Capture from camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if frame is None:
        print("Failed to load frame")
        exit()
    
    # Test feature extraction
    start_time = time.time()
    features = extractor.extract_features(frame)
    feature_time = time.time() - start_time
    
    if features is not None:
        print(f"Feature extraction took {feature_time:.2f} seconds")
        print(f"Feature shape: {features.shape}")
    
    # Test caption generation
    start_time = time.time()
    caption = extractor.generate_caption(frame)
    caption_time = time.time() - start_time
    
    print(f"Caption generation took {caption_time:.2f} seconds")
    print(f"Caption: {caption}")
    
    # Test question answering
    question = "What objects are visible in this image?"
    start_time = time.time()
    answer = extractor.answer_question(frame, question)
    answer_time = time.time() - start_time
    
    print(f"Question answering took {answer_time:.2f} seconds")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Display the image
    cv2.imshow("Test Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 