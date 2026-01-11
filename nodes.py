"""
NSFW Guard Nodes for ComfyUI

Provides NSFW content detection using YOLO ONNX model (packaged as .pt).
- Auto-downloads model and labels if missing
- Checks images and interrupts workflow if NSFW detected

Model: falconsai_yolov9_nsfw_model_quantized.pt (ONNX format)
"""

import os
import json
import numpy as np
import torch
from PIL import Image
import urllib.request
import ssl
import onnxruntime as ort

import folder_paths
from server import PromptServer
from nodes import interrupt_processing

# Constants
MODEL_URL = "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/main/falconsai_yolov9_nsfw_model_quantized.pt"
LABELS_URL = "https://huggingface.co/Falconsai/nsfw_image_detection/resolve/main/labels.json"
MODEL_FILENAME = "falconsai_yolov9_nsfw_model_quantized.pt"
LABELS_FILENAME = "labels.json"

class NSFWContentError(Exception):
    """Custom exception for NSFW content detection."""
    def __init__(self, prediction: str, confidence: float, threshold: float):
        self.error_type = "nsfw_content_detected"
        self.prediction = prediction
        self.confidence = confidence
        self.threshold = threshold
        message = f"NSFW content detected - workflow interrupted (prediction: {prediction}, confidence: {confidence:.2%}, threshold: {threshold:.2%})"
        super().__init__(message)
    
    def to_dict(self):
        return {
            "error": {
                "type": self.error_type,
                "message": str(self),
                "details": {
                    "prediction": self.prediction,
                    "confidence": self.confidence,
                    "threshold": self.threshold
                }
            }
        }

class NSFWCheck:
    """
    Checks images for NSFW content.
    Auto-downloads model if needed.
    """
    _session = None
    _labels = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "check_nsfw"
    CATEGORY = "safety"
    
    def _ensure_model(self):
        """Ensure model and labels are downloaded and loaded."""
        nsfw_folder = os.path.join(folder_paths.models_dir, "nsfw")
        if not os.path.exists(nsfw_folder):
            os.makedirs(nsfw_folder, exist_ok=True)
            
        model_path = os.path.join(nsfw_folder, MODEL_FILENAME)
        labels_path = os.path.join(nsfw_folder, LABELS_FILENAME)
        
        # Download model
        if not os.path.exists(model_path):
            print(f"[NSFW Guard] Downloading model to {model_path}...")
            self._download_file(MODEL_URL, model_path)
            
        # Download labels
        if not os.path.exists(labels_path):
            print(f"[NSFW Guard] Downloading labels to {labels_path}...")
            self._download_file(LABELS_URL, labels_path)
            
        # Load model if not loaded
        if NSFWCheck._session is None:
            print(f"[NSFW Guard] Loading model from {model_path}...")
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                NSFWCheck._session = ort.InferenceSession(model_path, providers=providers)
            except Exception as e:
                print(f"[NSFW Guard] Failed to load with CUDA, falling back to CPU: {e}")
                NSFWCheck._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                
        # Load labels if not loaded
        if NSFWCheck._labels is None:
            with open(labels_path, "r") as f:
                labels_map = json.load(f)
                # Ensure keys are integers
                NSFWCheck._labels = {int(k): v for k, v in labels_map.items()}

    def _download_file(self, url, path):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(url, context=ctx) as response:
                total_size = int(response.info().get('Content-Length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(path, 'wb') as f:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        
                        if total_size > 0:
                            percent = min(100, downloaded * 100 // total_size)
                            if downloaded % (block_size * 100) == 0: # Update every ~800KB
                                print(f"\r[NSFW Guard] Downloading: {percent}%", end="", flush=True)
                                
            print("\n[NSFW Guard] Download complete.")
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(f"Failed to download {url}: {e}")

    def check_nsfw(self, image: torch.Tensor, threshold: float = 0.5):
        self._ensure_model()
        
        session = NSFWCheck._session
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        batch_size = image.shape[0]
        
        max_nsfw_confidence = 0.0
        nsfw_detected = False
        
        for i in range(batch_size):
            # Preprocess: [H, W, C] -> Resize -> [1, C, H, W]
            img_tensor = image[i]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np, mode='RGB')
            
            # Resize to 224x224 (fixed for this model)
            img_resized = pil_image.resize((224, 224), Image.Resampling.BILINEAR)
            
            # Normalize and Transpose
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.transpose(img_array, (2, 0, 1)) # [C, H, W]
            input_tensor = np.expand_dims(img_array, axis=0).astype(np.float32)
            
            # Inference
            outputs = session.run([output_name], {input_name: input_tensor})
            predictions = outputs[0][0] # First batch item
            
            # Get scores
            # Assuming index 0 is 'normal' and index 1 is 'nsfw' based on labels.json
            
            # Apply sigmoid to normalize logits to 0-1
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
                
            normal_score = float(sigmoid(predictions[0]))
            nsfw_score = float(sigmoid(predictions[1]))
            
            # Track max NSFW confidence found in batch
            if nsfw_score > max_nsfw_confidence:
                max_nsfw_confidence = nsfw_score
            
            # Check threshold
            if nsfw_score >= threshold:
                nsfw_detected = True
                break
        
        if nsfw_detected:
            error = NSFWContentError(
                prediction="nsfw",
                confidence=max_nsfw_confidence,
                threshold=threshold
            )
            
            PromptServer.instance.send_sync("nsfw_guard.content_blocked", {
                "type": "nsfw_content_detected",
                "prediction": "nsfw",
                "confidence": max_nsfw_confidence,
                "threshold": threshold,
                "message": str(error)
            })
            
            interrupt_processing(True)
            raise Exception(json.dumps(error.to_dict()))
            
        return (image,)

NODE_CLASS_MAPPINGS = {
    "NSFWCheck": NSFWCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NSFWCheck": "NSFW Check (YOLO)",
}
