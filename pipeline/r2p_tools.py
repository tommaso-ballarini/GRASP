import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from transformers import CLIPModel, CLIPProcessor
from config import Config

# =============================================================================
# SEZIONE 1: CLIP SCORE CALCULATOR (Per Cross-Modal Attribute Verification - Eq. 5)
# =============================================================================

class ClipScoreCalculator:
    def __init__(self, device="cuda", model_name=None):
        self.device = device
        model_name = model_name or Config.Models.CLIP_MODEL_336
        print(f"Loading CLIP model: {model_name}...")
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()

    def format_text_for_clip(self, sentence, category):
        """
        Formats text into CLIP-compatible prompts.
        Original logic from R2P to maintain consistency.
        """
        # Se la frase è una lista di attributi separati da virgola
        return [f"A photo of a {category} with {item.strip()}" for item in sentence.split(',')]

    def get_clip_image_feature(self, image_path_or_obj):
        """Extracts and normalizes CLIP image features."""
        import torch.nn.functional as F
        if isinstance(image_path_or_obj, str):
            image = Image.open(image_path_or_obj).convert('RGB')
        else:
            image = image_path_or_obj.convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            output = self.clip_model.get_image_features(inputs['pixel_values'].to(self.device))
        # get_image_features può restituire un tensore o un oggetto — gestiamo entrambi
        if hasattr(output, 'image_embeds'):
            features = output.image_embeds
        elif hasattr(output, 'pooler_output'):
            features = output.pooler_output
        else:
            features = output
        return F.normalize(features, p=2, dim=-1)

    def get_text_features(self, formatted_sentences):
        """Extracts and normalizes CLIP text features."""
        import torch.nn.functional as F
        inputs = self.feature_extractor(text=formatted_sentences, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.clip_model.get_text_features(inputs["input_ids"], inputs["attention_mask"])
        if hasattr(output, 'text_embeds'):
            features = output.text_embeds
        elif hasattr(output, 'pooler_output'):
            features = output.pooler_output
        else:
            features = output
        return F.normalize(features, p=2, dim=-1)

    def compute_attribute_score(self, image, attribute_list):
        """
        Computes the average CLIP similarity score AND breakdown per attribute.
        """
        if not attribute_list:
            return 0.0, {}
            
        if len(attribute_list) > 20: # Aumentato limite per sicurezza
            attribute_list = attribute_list[:20]

        try:
            # 1. Get Text Embeddings
            text_features = self.get_text_features(attribute_list) 
            
            # 2. Get Image Embeddings
            image_features = self.get_clip_image_feature(image)

            # 3. Expand image features
            image_features = image_features.expand(text_features.size(0), -1)

            # 4. Compute Cosine Similarity
            similarities = torch.sum(image_features * text_features, dim=1)
            
            # 5. Prepare Breakdown
            # Convert tensor to list of floats
            scores_list = similarities.detach().cpu().tolist()
            
            # Create dictionary { "attribute text": 0.245 }
            breakdown = {attr: score for attr, score in zip(attribute_list, scores_list)}
            
            # Return Mean AND Breakdown
            return torch.mean(similarities).item(), breakdown
            
        except Exception as e:
            print(f"Error in CLIP calculation: {e}")
            return 0.0, {}
            
    

# =============================================================================
# SEZIONE 2: LLM-LEVEL CONFIDENCE EXTRACTOR (THE WORKING SOLUTION)
# =============================================================================

# from typing import Dict, Tuple, Optional

# class LLMConfidenceExtractor:
#     """
#     Extracts token-level confidence by intercepting model.llm.generate().
#     This is the PRODUCTION-READY solution for MiniCPM-o-2_6.
    
#     Works by:
#     1. Patching model.llm.generate() to force output_scores=True
#     2. Capturing the GenerateOutput before .chat() discards it
#     3. Extracting Yes/No probabilities from first-token logits
#     """
    
#     def __init__(self, model_interface, device="cuda"):
#         """
#         Args:
#             model_interface: The MiniCPMModel instance (reasoner.model_interface)
#         """
#         self.model = model_interface.model
#         self.tokenizer = model_interface.tokenizer
#         self.device = device
        
#         # Verify model structure
#         if not hasattr(self.model, 'llm'):
#             raise ValueError("Model does not have .llm attribute. This extractor is for MiniCPM-o models.")
        
#         self.llm = self.model.llm
        
#         # Get Yes/No token IDs (confirmed from diagnostic: Yes=9454, No=2753)
#         self.yes_id = self._get_token_id("Yes")
#         self.no_id = self._get_token_id("No")
        
#         print(f"🔧 LLMConfidenceExtractor initialized:")
#         print(f"   LLM type: {type(self.llm).__name__}")
#         print(f"   Yes token ID: {self.yes_id}")
#         print(f"   No token ID: {self.no_id}")
    
#     def _get_token_id(self, word: str) -> int:
#         """Get token ID from vocabulary."""
#         vocab = self.tokenizer.get_vocab()
#         if word in vocab:
#             return vocab[word]
#         # Fallback: encode
#         tokens = self.tokenizer.encode(word, add_special_tokens=False)
#         return tokens[0] if tokens else -1
    
#     def query_with_confidence(
#         self, 
#         image, 
#         prompt: str,
#         image2 = None,
#         max_new_tokens: int = 5
#     ) -> Tuple[Dict[str, float], str]:
#         """
#         Query the model and extract Yes/No confidence from logits.
        
#         Args:
#             image: Primary image (PIL.Image or path)
#             prompt: Text prompt (should ask for Yes/No answer)
#             image2: Optional second image for pairwise comparison
#             max_new_tokens: Max tokens to generate
            
#         Returns:
#             Tuple of (confidence_dict, response_text)
#         """
#         # Load images if paths
#         if isinstance(image, str):
#             image = Image.open(image).convert('RGB')
#         if image2 is not None and isinstance(image2, str):
#             image2 = Image.open(image2).convert('RGB')
        
#         # Build message
#         if image2 is not None:
#             content = [image, image2, prompt]
#         else:
#             content = [image, prompt]
        
#         msgs = [{"role": "user", "content": content}]
        
#         # Storage for captured output
#         captured = {"output": None}
        
#         # Store original generate
#         original_generate = self.llm.generate
        
#         def intercepting_generate(*args, **kwargs):
#             # Force score output
#             kwargs['output_scores'] = True
#             kwargs['return_dict_in_generate'] = True
#             result = original_generate(*args, **kwargs)
#             captured["output"] = result
#             return result
        
#         try:
#             # Patch
#             self.llm.generate = intercepting_generate
            
#             # Call chat normally
#             response_text = self.model.chat(
#                 msgs=msgs,
#                 tokenizer=self.tokenizer,
#                 max_new_tokens=max_new_tokens,
#                 sampling=False  # Greedy for consistency
#             )
            
#             # Handle OmniOutput or string
#             if hasattr(response_text, 'text'):
#                 response_text = response_text.text
#             elif not isinstance(response_text, str):
#                 response_text = str(response_text)
            
#             # Extract confidence from captured output
#             output = captured["output"]
            
#             if output is not None and hasattr(output, 'scores') and output.scores:
#                 confidence = self._extract_confidence_from_scores(output.scores)
#                 confidence['method'] = 'logits'
#                 confidence['response'] = response_text
#             else:
#                 # Fallback to text parsing
#                 confidence = self._text_fallback(response_text)
#                 confidence['response'] = response_text
            
#             return confidence, response_text
            
#         except Exception as e:
#             print(f"⚠️ LLMConfidenceExtractor error: {e}")
#             import traceback
#             traceback.print_exc()
#             return {
#                 'yes_confidence': 0.5,
#                 'no_confidence': 0.5,
#                 'method': 'error',
#                 'error': str(e)
#             }, f"Error: {e}"
            
#         finally:
#             # Always restore original
#             self.llm.generate = original_generate
    
    def _extract_confidence_from_scores(self, scores: tuple) -> Dict[str, float]:
        """
        Extract Yes/No confidence from the first token's logits.
        
        Args:
            scores: Tuple of tensors, one per generated token
        """
        # Get first token logits
        first_logits = scores[0]  # Shape: [batch_size, vocab_size]
        
        if first_logits.dim() > 1:
            first_logits = first_logits[0]  # [vocab_size]
        
        # Get full probability distribution
        probs = torch.softmax(first_logits, dim=0)
        
        # Extract Yes/No probabilities
        yes_prob = probs[self.yes_id].item() if self.yes_id >= 0 else 0.0
        no_prob = probs[self.no_id].item() if self.no_id >= 0 else 0.0
        
        # Also get raw logits for debugging
        yes_logit = first_logits[self.yes_id].item() if self.yes_id >= 0 else 0.0
        no_logit = first_logits[self.no_id].item() if self.no_id >= 0 else 0.0
        
        # Normalize to Yes/No space (binary)
        total = yes_prob + no_prob
        if total > 0:
            yes_norm = yes_prob / total
            no_norm = no_prob / total
        else:
            yes_norm = no_norm = 0.5
        
        return {
            'yes_confidence': yes_norm,
            'no_confidence': no_norm,
            'raw_yes_prob': yes_prob,
            'raw_no_prob': no_prob,
            'yes_logit': yes_logit,
            'no_logit': no_logit
        }
    
    def _text_fallback(self, response: str) -> Dict[str, float]:
        """
        Graduated text-based fallback when logits unavailable.
        More nuanced than binary 0.95/0.05.
        """
        if not response:
            return {'yes_confidence': 0.5, 'no_confidence': 0.5, 'method': 'text_empty'}
        
        response = response.lower().strip()
        words = response.split()
        first_word = words[0] if words else ""
        
        # Check for definitive first word
        if first_word in ["yes", "yes.", "yes,"]:
            return {'yes_confidence': 0.85, 'no_confidence': 0.15, 'method': 'text_definitive'}
        elif first_word in ["no", "no.", "no,"]:
            return {'yes_confidence': 0.15, 'no_confidence': 0.85, 'method': 'text_definitive'}
        
        # Check for presence anywhere in first 30 chars
        snippet = response[:30]
        if "yes" in snippet and "no" not in snippet:
            return {'yes_confidence': 0.70, 'no_confidence': 0.30, 'method': 'text_contains'}
        elif "no" in snippet and "yes" not in snippet:
            return {'yes_confidence': 0.30, 'no_confidence': 0.70, 'method': 'text_contains'}
        
        # Hedging language
        if any(w in response for w in ["partially", "somewhat", "maybe", "possibly", "unclear", "not sure"]):
            return {'yes_confidence': 0.50, 'no_confidence': 0.50, 'method': 'text_hedging'}
        
        # Default ambiguous
        return {'yes_confidence': 0.50, 'no_confidence': 0.50, 'method': 'text_ambiguous'}


# =============================================================================
# LEGACY: ConfidenceCalculator (kept for backwards compatibility)
# =============================================================================

class ConfidenceCalculator:
    """
    DEPRECATED: Use LLMConfidenceExtractor instead.
    Kept only for backwards compatibility with existing code.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.yes_id = 9454  # Hardcoded for MiniCPM
        self.no_id = 2753
        print("⚠️ ConfidenceCalculator is DEPRECATED. Use LLMConfidenceExtractor for reliable logit extraction.")

    def calculate_binary_confidence(self, outputs, response_text=None):
        """Fallback implementation - always uses text parsing."""
        return self._text_fallback(response_text)
    
    def _text_fallback(self, response_text):
        if not response_text:
            return {
                "yes_confidence": 0.5,
                "no_confidence": 0.5,
                "confidence_score": 0.5,
                "chosen_answer": "unknown",
                "method": "text_fallback_no_text"
            }
        
        text_lower = response_text.lower().strip()
        first_chars = text_lower[:10]
        
        if "yes" in first_chars:
            return {
                "yes_confidence": 0.85,
                "no_confidence": 0.15,
                "confidence_score": 0.85,
                "chosen_answer": "yes",
                "method": "text_fallback"
            }
        elif "no" in first_chars:
            return {
                "yes_confidence": 0.15,
                "no_confidence": 0.85,
                "confidence_score": 0.85,
                "chosen_answer": "no",
                "method": "text_fallback"
            }
        else:
            return {
                "yes_confidence": 0.5,
                "no_confidence": 0.5,
                "confidence_score": 0.5,
                "chosen_answer": "unknown",
                "method": "text_fallback_ambiguous"
            }
            
# =============================================================================
# SEZIONE 3: MODEL ADAPTERS (Per Formattazione Prompt Multi-Immagine)
# =============================================================================

class ModelAdapter:
    """Base class for model-specific message format adapters."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

# class MiniCPMAdapter(ModelAdapter):
#     """Adapter for MiniCPM model message format (v2.6 strict)."""

#     def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
#         """
#         Prepara il payload per il Pairwise Reasoning: 2 Immagini + Prompt.
#         """
#         # Load images if paths are provided
#         if isinstance(test_image, str):
#             test_image = Image.open(test_image).convert('RGB')
#         if isinstance(ret_image, str):
#             ret_image = Image.open(ret_image).convert('RGB')
            
#         # MiniCPM accetta una lista di oggetti nel content
#         return [{
#             'role': 'user', 
#             'content': [test_image, ret_image, prompt]
#         }]
    
#     def format_text_options_msgs(self, test_image, prompt):
#         """
#         Prepara il payload standard: 1 Immagine + Prompt.
#         """
#         if isinstance(test_image, str):
#             test_image = Image.open(test_image).convert('RGB')

#         return [{
#             'role': 'user', 
#             'content': [test_image, prompt]
#         }]
