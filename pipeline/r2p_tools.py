import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from config import Config


class ClipScoreCalculator:
    def __init__(self, device="cuda", model_name=None):
        self.device = device
        model_name = model_name or Config.Models.CLIP_MODEL_336
        print(f"Loading CLIP model: {model_name}...")
        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()

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
        # get_image_features can return either a tensor or an object; handle both
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
            
        if len(attribute_list) > 20: 
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
    

