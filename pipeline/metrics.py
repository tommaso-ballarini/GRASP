# pipeline/metrics.py
"""
Metriche di Valutazione per Image Personalization.
Supportate dalla letteratura:
- CLIP-I: Identity preservation via CLIP image embeddings
- CLIP-T: Text-image alignment via CLIP
- DINO-I: Identity preservation via DINO features (texture/details)
- TIFA: VQA-based attribute faithfulness

References:
- TIFA (Hu et al., ICCV 2023): VQA-based faithfulness evaluation
- CLIP (Radford et al., 2021): Vision-language similarity
- DINO (Caron et al., 2021): Self-supervised visual features
- DreamBooth (Ruiz et al., CVPR 2023): CLIP-I/DINO-I for personalization
"""

import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


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


class MetricsCalculator:
    """
    Calcola metriche di valutazione per image personalization.
    
    Lazy loading: i modelli vengono caricati solo quando necessario.
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
        """Lazy load DINO model."""
        if self._dino_model is None:
            print("   📦 Loading DINO model...")
            self._dino_model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vitl14'
            ).to(self.device).eval()
            from torchvision import transforms
            self._dino_processor = transforms.Compose([
                transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(518),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
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
        
        Used by DreamBooth and IP-Adapter papers for identity preservation.
        Higher = better identity preservation.
        
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
            
            gen_features = model.get_image_features(**gen_inputs)
            ref_features = model.get_image_features(**ref_inputs)
            
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
        
        Measures prompt faithfulness / text-image alignment.
        Higher = better prompt following.
        
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
            img_features = model.get_image_features(**img_inputs)
            
            # Get text features
            text_inputs = processor(text=prompt, return_tensors="pt", truncation=True).to(self.device)
            text_features = model.get_text_features(**text_inputs)
            
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
        
        DINO features are better than CLIP for fine-grained visual details
        (texture, material, small features). Used in DreamBooth evaluation.
        Higher = better identity preservation at detail level.
        
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
            # Process images
            gen_tensor = processor(generated_image).unsqueeze(0).to(self.device)
            ref_tensor = processor(reference_image).unsqueeze(0).to(self.device)
            
            # Get CLS token features
            gen_features = model(gen_tensor)
            ref_features = model(ref_tensor)
            
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
            
            # Generate question based on attribute type
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
