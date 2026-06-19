"""
pipeline/build_database.py

Builds the fingerprints database for the R2P-GEN pipeline (FLUX Edition).
Uses CLIP's Text Encoder to force selection of the image that clearly
shows structural details, readable text and logos, avoiding representation
smoothing of centroids.

Workflow:
    1. Discover all concepts in perva-data (train/category/concept_id/)
    2. Select the TOP-3 representative images via Text-Image Cosine Similarity
    3. Extract fingerprints with Qwen3-VL (structured JSON) from the TOP-1
    4. Save database.json containing concept_dict + path_to_concept (including
         backups for recovery)

"""

import os
import sys
import json
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config

# ---------------------------------------------------------------------------
# Constants / env
# ---------------------------------------------------------------------------

_DEFAULT_PERVA = "/leonardo_work/IscrC_MUSE/tballari/perva-data"
PERVA_DATA_DIR = os.environ.get("R2P_PERVA_DATA", _DEFAULT_PERVA)

# ---------------------------------------------------------------------------
# Prompts for Qwen3-VL
# ---------------------------------------------------------------------------

_ONESHOT_IMAGE_PATH = os.path.join(project_root, "example_database", "wnr.jpg")

_ANSWER_FORMAT = {
    "general":          "a brief description of the object in one sentence.",
    "category":         "category of the object",
    "shape":            "shape of the object",
    "material":         "material of the object",
    "color":            "color of the object",
    "pattern":          "any distinct pattern if present, else 'none'",
    "brand/text":       "any text or brand visible on the object, else 'none'",
    "distinct features":"any distinct feature that makes the object unique",
}

_ONESHOT_ANSWER = json.dumps({
    "general":          "<wnr> is a decorative ceramic plate with an elegant floral design around the rim.",
    "category":         "Plate",
    "shape":            "Round with slightly raised edges",
    "material":         "Ceramic",
    "color":            "White base with orange and blue flowers and green leaves on the border",
    "pattern":          "Floral pattern with small, evenly spaced blossoms and foliage",
    "brand/text":       "none",
    "distinct features":"Intricate detailing of flower motifs along the edge",
}, indent=2)


def _build_extraction_messages(
    image: Image.Image,
    category: str,
    concept_id: str,
) -> list:
    question_test = (
        f"Describe the {category} in the image identified by the concept-identifier "
        f"<{concept_id}> and highlight what makes it unique.\n"
        f"Your response MUST be valid JSON and follow EXACTLY this format:\n"
        f"{json.dumps(_ANSWER_FORMAT, indent=2)}\n\n"
        f"RULES:\n"
        f'- The "general" field MUST begin with "<{concept_id}> is ...".\n'
        f"- List only the most distinguishing features that set this object apart.\n"
        f"- Avoid generic descriptions that apply to every object in this category.\n"
        f"- Respond ONLY with the JSON object. No extra text, no markdown fences."
    )

    use_oneshot = os.path.exists(_ONESHOT_IMAGE_PATH)

    if use_oneshot:
        oneshot_img = Image.open(_ONESHOT_IMAGE_PATH).convert("RGB")
        question_example = (
            "Describe the plate in the image identified by the concept-identifier "
            "<wnr> and highlight what makes it unique.\n"
            f"Your response MUST follow EXACTLY the JSON format shown below:\n"
            f"{json.dumps(_ANSWER_FORMAT, indent=2)}\n\n"
            "RULES:\n"
            '- The "general" field MUST begin with "<wnr> is ...".\n'
            "- Respond ONLY with the JSON object. No extra text, no markdown fences."
        )
        msgs = [
            {"role": "user",      "content": [{"type": "image", "image": oneshot_img},
                                               {"type": "text",  "text": question_example}]},
            {"role": "assistant", "content": _ONESHOT_ANSWER},
            {"role": "user",      "content": [{"type": "image", "image": image},
                                               {"type": "text",  "text": question_test}]},
        ]
    else:
        msgs = [
            {"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text",  "text": question_test}]},
        ]

    return msgs


def _parse_json_response(raw: str) -> dict:
    cleaned = raw.strip().strip("```json").strip("```").strip()
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    else:
        raise ValueError(f"Nessun JSON trovato nella risposta: {raw[:200]}")
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# NEW SELECTOR: Semantic CLIP Selector (Text-Driven)
# ---------------------------------------------------------------------------

class _SemanticCLIPSelector:
    """
    Selects the most representative images by computing similarity
    between visual features and an ideal textual prompt (Text Prior).
    This addresses the "Representation Smoothing" problem of centroids.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        print("   📎 Loading Semantic CLIP for Top-K text-driven image selection...")
        self.model = CLIPModel.from_pretrained(Config.Models.CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(Config.Models.CLIP_MODEL)

    @torch.no_grad()
    def select(self, image_paths: list[str], category: str, k: int = 3) -> list[str]:
        """
        Returns the paths of the top-K images most similar to the ideal prompt.
        """
        if len(image_paths) == 1:
            return [image_paths[0]]

        # 1. Definition of the Text Prior (Semantic Magnet)
        text_prompt = (
            f"A clear frontal photo of a {category}, perfectly showing "
            "brand logos, readable text, and distinct structural features."
        )

        # 2. Load images
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # 3. Process mixed input (Text + Images)
        inputs = self.processor(
            text=[text_prompt], 
            images=images, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 4. Extract embeddings (joint model)
        outputs = self.model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalization
        import torch.nn.functional as F
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        # 5. Compute Cosine Similarity (Dot Product)
        similarities = (image_embeds @ text_embeds.T).squeeze()
        
        if similarities.dim() == 0:
            return [image_paths[0]]

        # 6. Select Top-K (Avoid errors if total images are less than K)
        actual_k = min(k, len(image_paths))
        top_scores, top_indices = torch.topk(similarities, k=actual_k)

        # Return ordered list of paths (highest first)
        return [image_paths[idx.item()] for idx in top_indices]

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DatabaseBuilder
# ---------------------------------------------------------------------------

class DatabaseBuilder:
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(
        self,
        perva_data_dir: str = PERVA_DATA_DIR,
        dataset_split: str = None,
        debug_mode: bool = None,
        debug_limit: int = None,
        use_clip_selection: bool = True,
        ignore_laion: bool = None,
        seed: int = None,
        device: str = None,
    ):
        self.perva_data_dir  = perva_data_dir
        self.dataset_split   = dataset_split   or Config.BuildDatabase.DATASET_SPLIT
        self.debug_mode      = debug_mode      if debug_mode is not None else Config.BuildDatabase.DEBUG_MODE
        self.debug_limit     = debug_limit     or Config.BuildDatabase.DEBUG_LIMIT
        self.use_clip_sel    = use_clip_selection
        self.ignore_laion    = ignore_laion    if ignore_laion is not None else Config.BuildDatabase.IGNORE_LAION
        self.seed            = seed            or Config.BuildDatabase.SEED
        self.device          = device          or Config.GPU.DEVICE

        if Config.Database.CANONICAL_NAME:
            out_name = "database.json"
        else:
            out_name = f"database_perva_{self.dataset_split}.json"
        self.output_path = os.path.join(Config.Paths.DATABASE_DIR, out_name)

        self._reasoner   = None
        self._clip_sel   = None

        self._database = {
            "concept_dict":   {},
            "path_to_concept": {},
        }

    @property
    def reasoner(self):
        if self._reasoner is None:
            print("   📦 Loading Qwen3-VL for fingerprint extraction...")
            from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
            self._reasoner = Qwen3VLReasoning(
                model_path=Config.Models.QWEN3_MODEL,
                device=self.device,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                seed=self.seed,
            )
            print("   ✅ Qwen3-VL loaded.")
        return self._reasoner

    @property
    def clip_selector(self):
        if self._clip_sel is None and self.use_clip_sel:
            self._clip_sel = _SemanticCLIPSelector(device=self.device)
        return self._clip_sel

    def _get_concepts(self) -> list[dict]:
        abs_path = os.path.abspath(self.perva_data_dir)
        print(f"   🔍 perva-data: {abs_path}")

        if not os.path.exists(abs_path):
            print(f"   ❌ Directory non trovata: {abs_path}")
            return []

        splits = ["train", "test"] if self.dataset_split == "all" else [self.dataset_split]
        concepts = []

        for split in splits:
            split_dir = os.path.join(abs_path, split)
            if not os.path.exists(split_dir):
                continue

            for category in sorted(os.listdir(split_dir)):
                cat_path = os.path.join(split_dir, category)
                if not os.path.isdir(cat_path):
                    continue

                for concept_id in sorted(os.listdir(cat_path)):
                    concept_path = os.path.join(cat_path, concept_id)
                    if not os.path.isdir(concept_path):
                        continue
                    if self.ignore_laion and concept_id.lower() == "laion":
                        continue

                    images = [
                        os.path.join(concept_path, f)
                        for f in sorted(os.listdir(concept_path))
                        if os.path.splitext(f)[1] in self.VALID_EXTENSIONS
                        and os.path.isfile(os.path.join(concept_path, f))
                    ]

                    if images:
                        concepts.append({
                            "category":    category,
                            "concept_id":  concept_id,
                            "concept_path": concept_path,
                            "images":      images,
                            "split":       split,
                        })

        return concepts

    def _extract_fingerprints(self, image, category, concept_id):
        msgs = _build_extraction_messages(image, category, concept_id)
        output = self.reasoner.model_interface.chat(msgs)
        if isinstance(output, tuple):
            _, raw_text = output
        else:
            raw_text = output.get("sequences", "")
        fingerprints = _parse_json_response(raw_text)
        return fingerprints

    def _process_concept(self, concept_data: dict) -> tuple[str, dict]:
        """
        Processa un singolo concept usando il nuovo Text-Driven Selector.
        """
        category   = concept_data["category"]
        concept_id = concept_data["concept_id"]
        images     = concept_data["images"]

        # 1. Selezione immagini Top-K guidata dal Testo
        if self.use_clip_sel and self.clip_selector is not None:
            top_k_images = self.clip_selector.select(images, category=category, k=3)
            representative = top_k_images[0] # La vincitrice assoluta
        else:
            top_k_images = [images[0]]
            representative = images[0]

        # Carica l'immagine vincitrice
        image = Image.open(representative).convert("RGB")

        # Estrai fingerprints dalla vincitrice
        fingerprints = self._extract_fingerprints(image, category, concept_id)

        concept_key = f"<{concept_id}>"
        entry = {
            "name":     concept_id,
            "image":    images,             # tutte le immagini del concept
            "representative_image": representative,
            "top_k_images": top_k_images,   # NUOVO CAMPO: storicizza i backup per il recovery
            "info":     fingerprints,       # fingerprints JSON
            "category": category,
        }

        return concept_key, entry

    def build(self) -> dict:
        print("\n" + "="*70)
        print("BUILD DATABASE — R2P-GEN FLUX Edition (DIEGO VARIANTE SEMANTICA)")
        print("="*70)
        Config.print_summary()
        print(f"  perva-data   : {self.perva_data_dir}")
        print(f"  split        : {self.dataset_split}")
        print(f"  SemanticCLIP : {self.use_clip_sel}")
        print(f"  debug        : {self.debug_mode} (limit={self.debug_limit})")
        print(f"  output       : {self.output_path}")
        print("="*70 + "\n")

        print("[1/3] Discovering concepts...")
        all_concepts = self._get_concepts()
        if not all_concepts:
            print("❌ Nessun concept trovato. Controlla R2P_PERVA_DATA.")
            return {"success_count": 0, "total_concepts": 0}

        print(f"   Trovati {len(all_concepts)} concepts.")

        if self.debug_mode:
            all_concepts = all_concepts[:self.debug_limit]
            print(f"   DEBUG MODE: processing solo {len(all_concepts)} concepts.")

        print("\n[2/3] Extracting fingerprints (Qwen3-VL)...")
        success = 0
        for concept_data in tqdm(all_concepts, desc="Concepts"):
            try:
                key, entry = self._process_concept(concept_data)
                self._database["concept_dict"][key] = entry
                for img_path in entry["image"]:
                    self._database["path_to_concept"][img_path] = key
                success += 1
            except Exception as e:
                cid = concept_data.get("concept_id", "?")
                print(f"\n   ⚠️  Errore su '{cid}': {e}")
                continue

        print(f"\n[3/3] Saving database → {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self._database, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Done! {success}/{len(all_concepts)} concepts processati.")
        print(f"   Database: {os.path.abspath(self.output_path)}")

        if self._reasoner is not None:
            del self._reasoner
            self._reasoner = None
        if self._clip_sel is not None:
            self._clip_sel.cleanup()
            self._clip_sel = None
        torch.cuda.empty_cache()

        return {
            "success_count":  success,
            "total_concepts": len(all_concepts),
            "database_path":  os.path.abspath(self.output_path),
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build R2P-GEN fingerprint database (Semantic Variant)")
    parser.add_argument("--perva-data",  type=str, default=None,
                        help="Path a perva-data (default: R2P_PERVA_DATA env var)")
    parser.add_argument("--split",       type=str, default=None,
                        choices=["train", "test", "all"],
                        help="Dataset split")
    parser.add_argument("--debug",       action="store_true", default=None,
                        help="Debug mode: processa solo i primi N concepts")
    parser.add_argument("--debug-limit", type=int, default=None,
                        help="Numero concepts in debug mode")
    parser.add_argument("--no-clip",     action="store_true",
                        help="Disabilita Semantic CLIP selection")
    args = parser.parse_args()

    builder = DatabaseBuilder(
        perva_data_dir    = args.perva_data or PERVA_DATA_DIR,
        dataset_split     = args.split,
        debug_mode        = args.debug,
        debug_limit       = args.debug_limit,
        use_clip_selection= not args.no_clip,
    )
    builder.build()