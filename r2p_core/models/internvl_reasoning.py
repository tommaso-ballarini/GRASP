"""
r2p_core/models/internvl_reasoning.py

Wrapper InternVL3_5-8B per pipeline/judge.py.
Drop-in replacement per QwenReasoning nel ruolo di Final Judge.

Struttura speculare a qwen3_vl_reasoning.py:
  - InternVL3_5Model(ModelInterface)  → gestisce load + chat()
  - InternVL3_5Reasoning              → espone .adapter, .model_interface, .conf_calculator
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional

# InternVL3_5 richiede transformers standard + torchvision
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from r2p_core.models.model_interface import ModelInterface
from r2p_core.models.model_adapters import InternVLAdapter
from r2p_core.models.confidence_calculator import ConfidenceCalculator


# ---------------------------------------------------------------------------
# Costanti InternVL3_5 (da documentazione ufficiale)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

INTERNVL3_5_8B_PATH = (
    "/leonardo_work/IscrC_MUSE/tballari/models_cache/"
    "huggingface/InternVL3_5-8B"
)


# ---------------------------------------------------------------------------
# Preprocessing immagine (InternVL3_5 dynamic tiling)
# ---------------------------------------------------------------------------

def _build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
) -> list[Image.Image]:
    """
    Suddivide l'immagine in tile dinamici secondo il protocollo InternVL3_5.
    Aggiunge sempre una thumbnail dell'immagine intera come tile finale.
    """
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h

    # Trova la griglia (target_w x target_h) che minimizza il ritaglio
    best = None
    best_ratio_diff = float("inf")

    for n in range(min_num, max_num + 1):
        for rows in range(1, n + 1):
            cols = n // rows
            if rows * cols != n:
                continue
            ratio = (cols / rows)
            diff = abs(aspect - ratio)
            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best = (cols, rows)

    cols, rows = best
    tile_w = image_size * cols
    tile_h = image_size * rows

    resized = image.resize((tile_w, tile_h), Image.BICUBIC)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            box = (
                c * image_size, r * image_size,
                (c + 1) * image_size, (r + 1) * image_size,
            )
            tiles.append(resized.crop(box))

    # Thumbnail globale sempre presente
    thumbnail = image.resize((image_size, image_size), Image.BICUBIC)
    tiles.append(thumbnail)

    return tiles


def _load_image_tensor(
    image: Image.Image | str,
    max_num: int = 6,
    image_size: int = 448,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Converte una PIL Image (o path) nel tensore pixel_values atteso da InternVL3_5.
    Shape: (num_tiles, 3, image_size, image_size)
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    tiles = _dynamic_preprocess(image, max_num=max_num, image_size=image_size)
    transform = _build_transform(image_size)
    tensors = [transform(t) for t in tiles]
    return torch.stack(tensors).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# InternVL3_5Model
# ---------------------------------------------------------------------------

class InternVL3_5Model(ModelInterface):
    """
    ModelInterface per InternVL3_5-8B.

    Caricamento con split manuale dei layer su multi-GPU
    (InternVL3_5-8B non supporta device_map='auto' out-of-the-box).
    chat() restituisce {"sequences": str, "logits": Tensor | None}
    identico a Qwen3VLModel.
    """

    def __init__(
        self,
        model_path: str = INTERNVL3_5_8B_PATH,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        **kwargs,
    ):
        # Non chiamiamo super().__init__() perché ModelInterface
        # fa branch su model_path — InternVL3_5 ha un proprio loader.
        self.device = device
        self.model_path = model_path
        self._dtype = torch_dtype
        self._model_type = "internvl3_5"

        print(f"   📦 Loading InternVL3_5-8B from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch_dtype,
            trust_remote_code=True,
            # Flash Attention 2 se disponibile, altrimenti sdpa
            attn_implementation=attn_implementation,
            # device_map split automatico: funziona con InternVL3_5-8B
            # su singola GPU o multi-GPU con accelerate installato.
            device_map="auto",
        ).eval()

        # Esposto per compatibilità con codice che legge .processor
        self.processor = self.tokenizer
        self.image_processor = None

        print("   ✅ InternVL3_5-8B loaded.")

    # ------------------------------------------------------------------
    # chat()
    # ------------------------------------------------------------------

    def chat(self, msgs: list) -> dict:
        """
        Interfaccia unificata chat.

        Args:
            msgs: Lista messaggi in formato InternVLAdapter, es.:
                  [{"role": "user", "content": [
                      {"type": "image", "image": <PIL.Image>},
                      {"type": "text",  "text": "<image>\\nDomanda?"}
                  ]}]

        Returns:
            {"sequences": str, "logits": torch.Tensor | None}
        """
        # --- Estrai immagini e testo dal content strutturato ---
        images_pil: list[Image.Image] = []
        text_parts: list[str] = []

        user_content = msgs[0]["content"]   # InternVL2 è sempre single-turn
        for item in user_content:
            if item["type"] == "image":
                img = item["image"]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                images_pil.append(img)
            elif item["type"] == "text":
                # Rimuovi eventuali <image>\n già presenti nel testo
                # (InternVLAdapter li aggiunge, ma li riscriviamo noi)
                clean = item["text"].replace("<image>\n", "").strip()
                text_parts.append(clean)

        prompt_text = " ".join(text_parts)

        # --- Costruisci il prompt con i <image> token nella posizione corretta ---
        # InternVL2 si aspetta: "<image>\n" * num_images + domanda
        num_images = len(images_pil)
        image_tokens = "<image>\n" * num_images
        full_prompt = image_tokens + prompt_text

        # --- Tokenizza ---
        model_inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
        )
        input_ids = model_inputs.input_ids.to(self.device)
        attention_mask = model_inputs.attention_mask.to(self.device)

        # --- Prepara pixel_values ---
        # InternVL2 vuole un unico tensore concatenato di tutti i tile
        # di tutte le immagini, più num_patches_list per sapere quanti
        # tile appartengono a ciascuna immagine.
        pixel_values_list = []
        num_patches_list = []

        # Determina il device reale del modello (potrebbe essere su più GPU)
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype

        for img_pil in images_pil:
            pv = _load_image_tensor(
                img_pil,
                device=model_device,
                dtype=model_dtype,
            )
            pixel_values_list.append(pv)
            num_patches_list.append(pv.shape[0])

        pixel_values = (
            torch.cat(pixel_values_list, dim=0)
            if pixel_values_list
            else None
        )

        # --- Generazione ---
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        with torch.no_grad():
            output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_patches_list=num_patches_list if num_images > 0 else None,
                **generation_config,
            )

        # --- Decodifica (solo token nuovi) ---
        prompt_len = input_ids.shape[1]
        generated_ids = output.sequences[:, prompt_len:]
        decoded = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        last_logits = output.scores[-1][0] if output.scores else None

        return {"sequences": decoded, "logits": last_logits}


# ---------------------------------------------------------------------------
# InternVL3_5Reasoning
# ---------------------------------------------------------------------------

class InternVL3_5Reasoning:
    """
    Drop-in replacement per QwenReasoning nel ruolo di Final Judge.

    Espone la stessa API pubblica:
      .adapter          → InternVLAdapter
      .model_interface  → InternVL3_5Model
      .conf_calculator  → ConfidenceCalculator

    Usage in judge.py:
        from r2p_core.models.internvl_reasoning import InternVL3_5Reasoning
        reasoner = InternVL3_5Reasoning(model_path=..., device="cuda")
    """

    def __init__(
        self,
        model_path: str = INTERNVL3_5_8B_PATH,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        seed: Optional[int] = None,
        **kwargs,
    ):
        if seed is not None:
            torch.manual_seed(seed)

        self.model_interface = InternVL3_5Model(
            model_path=model_path,
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        self.adapter = InternVLAdapter(
            tokenizer=self.model_interface.tokenizer
        )

        self.conf_calculator = ConfidenceCalculator(
            tokenizer=self.model_interface.tokenizer
        )

    # ------------------------------------------------------------------
    # _resize — fix bug PIL assi invertiti (speculare a Qwen3VLReasoning)
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(image: Image.Image, max_dim: int = 896) -> Image.Image:
        """
        Ridimensiona mantenendo l'aspect ratio.
        Corregge il bug PIL: Image.size → (width, height),
        quindi max() va fatto su (w, h), non su shape numpy.
        """
        w, h = image.size
        if max(w, h) <= max_dim:
            return image
        scale = max_dim / max(w, h)
        return image.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS,
        )

    def cleanup(self):
        """Libera la VRAM."""
        if hasattr(self, "model_interface") and self.model_interface.model is not None:
            del self.model_interface.model
            self.model_interface.model = None
        torch.cuda.empty_cache()
        print("   🧹 InternVL3_5Reasoning resources released.")