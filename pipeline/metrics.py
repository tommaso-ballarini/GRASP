# pipeline/metrics.py
"""
Metrics for evaluating image personalization quality:
- CLIP-I: Identity preservation via CLIP image embeddings
- CLIP-T: Text-image alignment via CLIP
- DINO-I: Identity preservation via DINO features (texture/details)
- TIFA: VQA-based attribute faithfulness

"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config


@dataclass
class MetricsResult:
    """Container for all evaluation metrics."""
    clip_i: float = 0.0          # CLIP Image similarity (identity)
    clip_t: float = 0.0          # CLIP Text-image alignment
    dino_i: float = 0.0          # DINO identity similarity
    tifa_score: float = 0.0      # VQA attribute accuracy
    tifa_details: Dict = None    # Per-attribute VQA results
    final_score: float = 0.0     # Weighted aggregate
    
    def to_dict(self) -> Dict:
        return {
            "clip_i": self.clip_i,
            "clip_t": self.clip_t,
            "dino_i": self.dino_i,
            "tifa_score": self.tifa_score,
            "tifa_details": self.tifa_details or {},
            "final_score": self.final_score
        }

def _extract_features(output):
    """
    Extract features from the output of a model, handling different possible return types.
    """

    if torch.is_tensor(output):
        return output
    if hasattr(output, "image_embeds"):
        return output.image_embeds
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]  # fallback: CLS token
    raise TypeError(f"Impossibile estrarre le feature da {type(output)}")

class MetricsCalculator:
    """
    Class to compute evaluation metrics for image personalization.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Lazy-loaded models
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._dino_processor = None
        
        # Default weights for final score (literature-based)
        self.weights = {
            "clip_i": 0.25,    # Identity preservation
            "clip_t": 0.20,    # Prompt faithfulness
            "dino_i": 0.25,    # Fine-grained identity
            "tifa": 0.30       # Attribute accuracy (most interpretable)
        }
    
    # ========================================================================
    # LAZY LOADING
    # ========================================================================
    
    @property
    def clip_model(self):
        """Lazy load CLIP model."""
        if self._clip_model is None:
            print("   📦 Loading CLIP model...")
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(Config.Models.CLIP_MODEL).to(self.device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(Config.Models.CLIP_MODEL)
        return self._clip_model, self._clip_processor
    
    @property
    def dino_model(self):
        """Lazy load DINO model (via transformers, not torch.hub: torch.hub
        contacts GitHub directly and ignores HF_HUB_OFFLINE, so it fails on
        offline nodes even when the HF cache is already populated)."""
        if self._dino_model is None:
            print("   📦 Loading DINO model...")
            from transformers import AutoModel, AutoImageProcessor
            dino_id = getattr(Config.Models, "DINO_MODEL", "facebook/dinov2-large")
            self._dino_model = AutoModel.from_pretrained(dino_id).to(self.device).eval()
            self._dino_processor = AutoImageProcessor.from_pretrained(dino_id)
        return self._dino_model, self._dino_processor
    
    # ========================================================================
    # CLIP METRICS
    # ========================================================================
    
    def compute_clip_i(
        self, 
        generated_image: Union[str, Image.Image], 
        reference_image: Union[str, Image.Image]
    ) -> float:
        """
        CLIP-I: Cosine similarity between CLIP embeddings of generated and reference images.
    
        Args:
            generated_image: Generated image path or PIL Image
            reference_image: Reference/target image path or PIL Image
            
        Returns:
            float: Cosine similarity in [0, 1]
        """
        model, processor = self.clip_model
        
        # Load images
        if isinstance(generated_image, str):
            generated_image = Image.open(generated_image).convert("RGB")
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        
        with torch.no_grad():
            # Get image features
            gen_inputs = processor(images=generated_image, return_tensors="pt").to(self.device)
            ref_inputs = processor(images=reference_image, return_tensors="pt").to(self.device)
            
            gen_features = _extract_features(model.get_image_features(**gen_inputs))
            ref_features = _extract_features(model.get_image_features(**ref_inputs))
            
            # Normalize
            gen_features = F.normalize(gen_features, p=2, dim=-1)
            ref_features = F.normalize(ref_features, p=2, dim=-1)
            
            # Cosine similarity
            similarity = (gen_features @ ref_features.T).item()
        
        return max(0.0, similarity)  # Clamp to [0, 1]
    
    def compute_clip_t(
        self, 
        generated_image: Union[str, Image.Image], 
        prompt: str
    ) -> float:
        """
        CLIP-T: Cosine similarity between generated image and text prompt.
        
        Args:
            generated_image: Generated image path or PIL Image
            prompt: SDXL prompt used for generation
            
        Returns:
            float: Cosine similarity in [0, 1]
        """
        model, processor = self.clip_model
        
        if isinstance(generated_image, str):
            generated_image = Image.open(generated_image).convert("RGB")
        
        with torch.no_grad():
            # Get image features
            img_inputs = processor(images=generated_image, return_tensors="pt").to(self.device)
            img_features = _extract_features(model.get_image_features(**img_inputs))

            
            # Get text features
            text_inputs = processor(text=prompt, return_tensors="pt", truncation=True).to(self.device)
            text_features = _extract_features(model.get_text_features(**text_inputs))

            
            # Normalize
            img_features = F.normalize(img_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            # Cosine similarity
            similarity = (img_features @ text_features.T).item()
        
        return max(0.0, similarity)
    
    # ========================================================================
    # DINO METRICS
    # ========================================================================
    
    def compute_dino_i(
        self, 
        generated_image: Union[str, Image.Image], 
        reference_image: Union[str, Image.Image]
    ) -> float:
        """
        DINO-I: Cosine similarity between DINOv2 features.
        
        Args:
            generated_image: Generated image path or PIL Image
            reference_image: Reference/target image path or PIL Image
            
        Returns:
            float: Cosine similarity in [0, 1]
        """
        model, processor = self.dino_model
        
        # Load images
        if isinstance(generated_image, str):
            generated_image = Image.open(generated_image).convert("RGB")
        if isinstance(reference_image, str):
            reference_image = Image.open(reference_image).convert("RGB")
        
        with torch.no_grad():
            gen_inputs = processor(images=generated_image, return_tensors="pt").to(self.device)
            ref_inputs = processor(images=reference_image, return_tensors="pt").to(self.device)

            gen_features = model(**gen_inputs).pooler_output
            ref_features = model(**ref_inputs).pooler_output
            
            # Normalize
            gen_features = F.normalize(gen_features, p=2, dim=-1)
            ref_features = F.normalize(ref_features, p=2, dim=-1)
            
            # Cosine similarity
            similarity = (gen_features @ ref_features.T).item()
        
        return max(0.0, similarity)
    
    # ========================================================================
    # TIFA-STYLE VQA METRICS (placeholder - requires VLM)
    # ========================================================================
    
    def generate_tifa_questions(self, fingerprints: Dict) -> List[Dict]:
        """
        Generate VQA questions from fingerprints (TIFA-style).
        
        For each attribute, generate a binary question.
        
        Args:
            fingerprints: Dict of attribute -> value
            
        Returns:
            List of {"attribute": str, "question": str, "expected": str}
        """
        questions = []
        
        # Skip non-verifiable fields
        skip_fields = {"description", "sdxl_prompt", "category"}
        
        for attr, value in fingerprints.items():
            if attr in skip_fields or not value:
                continue
            
            # Skip negative/invalid values
            value_lower = str(value).lower()
            if any(neg in value_lower for neg in ["none", "n/a", "unknown", "no visible"]):
                continue
            
            # Generate a question based on the attribute type
            if attr == "color":
                question = f"Is the main color of the object {value}?"
            elif attr == "material":
                question = f"Is the object made of {value}?"
            elif attr == "shape":
                question = f"Does the object have a {value} shape?"
            elif attr == "pattern":
                question = f"Does the object have a {value} pattern?"
            elif attr in ["brand/text", "brand", "text"]:
                question = f"Is there visible text or logo showing '{value}'?"
            elif attr == "distinct features":
                question = f"Does the object show: {value}?"
            else:
                question = f"Does the image show {attr}: {value}?"
            
            questions.append({
                "attribute": attr,
                "question": question,
                "expected": value
            })
        
        return questions
    
    # ========================================================================
    # AGGREGATION
    # ========================================================================
    
    def compute_final_score(self, metrics: MetricsResult) -> float:
        """
        Compute weighted final score from all metrics.
        
        Formula: Σ(weight_i * metric_i) for all enabled metrics
        """
        score = 0.0
        total_weight = 0.0
        
        if metrics.clip_i > 0:
            score += self.weights["clip_i"] * metrics.clip_i
            total_weight += self.weights["clip_i"]
        
        if metrics.clip_t > 0:
            score += self.weights["clip_t"] * metrics.clip_t
            total_weight += self.weights["clip_t"]
        
        if metrics.dino_i > 0:
            score += self.weights["dino_i"] * metrics.dino_i
            total_weight += self.weights["dino_i"]
        
        if metrics.tifa_score > 0:
            score += self.weights["tifa"] * metrics.tifa_score
            total_weight += self.weights["tifa"]
        
        # Normalize by total weight used
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def cleanup(self):
        """Release GPU memory."""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_processor
            self._clip_model = None
            self._clip_processor = None
        
        if self._dino_model is not None:
            del self._dino_model
            del self._dino_processor
            self._dino_model = None
            self._dino_processor = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_evaluate(
    generated_image: Union[str, Image.Image],
    reference_image: Union[str, Image.Image],
    prompt: str = None,
    device: str = "cuda"
) -> Dict:
    """
    Quick evaluation with CLIP-I and optionally CLIP-T.
    
    Lightweight evaluation without loading DINO or VLM.
    """
    calc = MetricsCalculator(device=device)
    
    result = MetricsResult()
    result.clip_i = calc.compute_clip_i(generated_image, reference_image)
    
    if prompt:
        result.clip_t = calc.compute_clip_t(generated_image, prompt)
    
    result.final_score = calc.compute_final_score(result)
    
    calc.cleanup()
    
    return result.to_dict()
