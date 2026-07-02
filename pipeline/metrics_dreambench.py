"""
pipeline/metrics_dreambench.py

Metriche ufficiali per la Fase 4 (tabella del paper), DELIBERATAMENTE
separate da pipeline/r2p_tools.py::ClipScoreCalculator (usato nel
recovery loop di verify/refine) e da pipeline/metrics.py::MetricsCalculator
(usato per altri scopi interni, es. ablation).

Motivo della separazione: il recovery loop usa clip-vit-large-patch14-336
con un attribute-score per-fingerprint (non è CLIP-I/CLIP-T in senso
stretto). Qui invece replichiamo ESATTAMENTE il protocollo DreamBooth/
DreamBench:
    - CLIP-I / CLIP-T: openai/clip-vit-base-patch32
    - DINO:            facebook/dino-vits16 
calcolati come similarity diretta image-image (CLIP-I, DINO) o
image-text (CLIP-T), media pairwise su TUTTE le immagini reali del
soggetto (non una sola reference), come da paper originale.
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Union

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config


def _load_image(img: Union[str, Image.Image]) -> Image.Image:
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    return img.convert("RGB")


class ClipDreamBench:
    """CLIP-I e CLIP-T col checkpoint ufficiale ViT-B/32 (ImageNet-style
    cosine similarity diretta, NESSUN attribute-score, NESSUNA logica
    di accept/reject — solo numeri da riportare in tabella)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        model_id = getattr(Config.Models, "CLIP_DREAMBENCH_MODEL", "openai/clip-vit-base-patch32")
        print(f"   📦 Loading CLIP-DreamBench: {model_id}")
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)

    @torch.no_grad()
    def _image_features(self, img: Union[str, Image.Image]) -> torch.Tensor:
        img = _load_image(img)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model.vision_model(**inputs)
        feats = self.model.visual_projection(out.pooler_output)  # 768 → 512
        return F.normalize(feats, p=2, dim=-1)

    @torch.no_grad()
    def _text_features(self, text: str) -> torch.Tensor:
        inputs = self.processor(text=text, return_tensors="pt", truncation=True).to(self.device)
        out = self.model.text_model(**inputs)
        feats = self.model.text_projection(out.pooler_output)    # 512 → 512
        return F.normalize(feats, p=2, dim=-1)

    def clip_i(self, gen_image: Union[str, Image.Image], real_images: List[str]) -> float:
        """Media pairwise tra l'immagine generata e TUTTE le immagini reali
        del soggetto (protocollo ufficiale, non una sola reference)."""
        gen_feat = self._image_features(gen_image)
        sims = []
        for real_path in real_images:
            real_feat = self._image_features(real_path)
            sims.append((gen_feat @ real_feat.T).item())
        return sum(sims) / len(sims) if sims else 0.0

    def clip_t(self, gen_image: Union[str, Image.Image], prompt: str) -> float:
        gen_feat = self._image_features(gen_image)
        text_feat = self._text_features(prompt)
        return (gen_feat @ text_feat.T).item()

    def cleanup(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DinoDreamBench:
    """DINO-I col checkpoint ORIGINALE del paper DreamBooth: DINOv1 ViT-S/16
    (facebook/dino-vits16, Caron et al. 2021). Stesso checkpoint usato in
    BLIP-Diffusion, ComFusion, DreamMatcher e nella stragrande maggioranza
    della letteratura subject-driven generation comparabile.

    NB: architettura ViT "vanilla" di transformers, NON Dinov2Model — va
    caricata con ViTModel/ViTImageProcessor, non con AutoModel/AutoImageProcessor
    (che per un repo facebook/dino-vits16 caricherebbero comunque la classe
    giusta in automatico, ma qui siamo espliciti per evitare ambiguità se in
    futuro si volesse swappare con una variante DINOv2)."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        model_id = getattr(Config.Models, "DINO_DREAMBENCH_MODEL", "facebook/dino-vits16")
        print(f"   📦 Loading DINO-DreamBench: {model_id}")
        from transformers import ViTModel, ViTImageProcessor
        self.model = ViTModel.from_pretrained(
            model_id,
            add_pooling_layer=False,
            local_files_only=True,
        ).to(device).eval()
        self.processor = ViTImageProcessor.from_pretrained(   # <-- mancava
            model_id,
            local_files_only=True,
        )

    @torch.no_grad()
    def _features(self, img):
        img = _load_image(img)
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        cls = out.hidden_states[-1][:, 0, :]
        return F.normalize(cls, p=2, dim=-1)

    def dino_i(self, gen_image: Union[str, Image.Image], real_images: List[str]) -> float:
        gen_feat = self._features(gen_image)
        sims = []
        for real_path in real_images:
            real_feat = self._features(real_path)
            sims.append((gen_feat @ real_feat.T).item())
        return sum(sims) / len(sims) if sims else 0.0

    def cleanup(self):
        del self.model, self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()