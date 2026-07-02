"""
pipeline/build_dataset.py

Builds the fingerprints database for the R2P-GEN pipeline (FLUX Edition).

Workflow:
    1. Discover all concepts in perva-data (train/category/concept_id/)
    2. Select the representative image via CLIP centroid
    3. Extract fingerprints with Qwen3-VL (structured JSON)
    4. Save database.json with concept_dict + path_to_concept

The flux_prompt is NOT generated here: it is built on the fly in flux_loop.py
via build_flux_prompt(fingerprints, target_context), which is deterministic
and depends on the target context (it may vary across runs).

Environment variables:
    R2P_PERVA_DATA   → path to the perva-data folder
                                         default: 
    R2P_MODELS_BASE  → base path for HuggingFace models
                                         default: uses repo-id (automatic download)
    R2P_CLUSTER_MODE → "true" to enable cluster mode in config.py
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

# perva-data path: env var > Leonardo cluster fallback
_DEFAULT_PERVA = "/leonardo_work/IscrC_MUSE/tballari/perva-data"
PERVA_DATA_DIR = os.environ.get("R2P_PERVA_DATA", _DEFAULT_PERVA)


# ---------------------------------------------------------------------------
# Qwen3-VL prompt (adapted from R2P get_detailed_input_msgs_household)
# ---------------------------------------------------------------------------

# One-shot example: same object used in the original R2P repo (wnr = ceramic plate).
# If the image is unavailable, the system automatically falls back to zero-shot.
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
    """
    Builds the Qwen3-VL messages to extract fingerprints in JSON.

    Uses one-shot if the example image is available, zero-shot otherwise.

    Args:
        image:      PIL Image dell'oggetto da analizzare
        category:   categoria rilevata via CLIP (es. "bag")
        concept_id: identificatore unico del concetto (es. "alx")
    """
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
    """
    Cleans and parses the Qwen3-VL JSON response.
    Handles markdown fences and extra text around the JSON.
    """
    # Remove markdown fences
    cleaned = raw.strip().strip("```json").strip("```").strip()

    # Estrai solo il blocco JSON se c'è testo extra
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    else:
        raise ValueError(f"Nessun JSON trovato nella risposta: {raw[:200]}")

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# CLIP image selector (kept from the original build_database)
# ---------------------------------------------------------------------------

class _CLIPSelector:
    """Select the most representative image of a concept via CLIP centroid."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        print("   📎 Loading CLIP for image selection...")
        self.model = CLIPModel.from_pretrained(Config.Models.CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(Config.Models.CLIP_MODEL)

    @torch.no_grad()
    def select(self, image_paths: list[str], seed: int = 42, top_k: int = 3) -> tuple[str, list[str]]:
        """
        Returns the path of the image closest to the CLIP centroid and the top-K list.
        """
        if len(image_paths) == 1:
            return image_paths[0], [image_paths[0]]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.get_image_features(**inputs)
        
        if hasattr(output, 'image_embeds'):
            features = output.image_embeds
        elif hasattr(output, 'pooler_output'):
            features = output.pooler_output
        else:
            features = output

        import torch.nn.functional as F
        features = F.normalize(features, p=2, dim=-1)

        centroid = features.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=-1)

        similarities = (features @ centroid.T).squeeze()
        
        # FIX ABLAZIONI: Prendi i top K indici ordinati
        k = min(top_k, len(image_paths))
        top_indices = similarities.argsort(descending=True)[:k].tolist()
        
        best_path = image_paths[top_indices[0]]
        top_k_paths = [image_paths[i] for i in top_indices]

        return best_path, top_k_paths
    
    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DatabaseBuilder
# ---------------------------------------------------------------------------

class DatabaseBuilder:
    """
    Builds the fingerprints database for R2P-GEN.

    Per ogni concept:
    1. Select the representative image (CLIP centroid)
      2. Extracts fingerprints with Qwen3-VL
        3. Save in database.json (including the representative image path,
    """

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

        # Output path
        if Config.Database.CANONICAL_NAME:
            out_name = "database_centroid.json"
        else:
            out_name = f"database_centroid_perva_{self.dataset_split}.json"
        self.output_path = os.path.join(Config.Paths.DATABASE_DIR, out_name)

        # Lazy loaded
        self._reasoner   = None
        self._clip_sel   = None

        self._database = {
            "concept_dict":   {},
            "path_to_concept": {},
        }

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

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
            self._clip_sel = _CLIPSelector(device=self.device)
        return self._clip_sel

    # ------------------------------------------------------------------
    # Dataset discovery
    # ------------------------------------------------------------------

    def _get_concepts(self) -> list[dict]:
        """Scan perva-data and return the list of all concepts."""
        abs_path = os.path.abspath(self.perva_data_dir)
        print(f"   🔍 perva-data: {abs_path}")

        if not os.path.exists(abs_path):
            print(f"   ❌ Directory not found: {abs_path}")
            print(f"      Set R2P_PERVA_DATA to override the path.")
            return []

        splits = ["train", "test"] if self.dataset_split == "all" else [self.dataset_split]
        concepts = []

        for split in splits:
            split_dir = os.path.join(abs_path, split)
            if not os.path.exists(split_dir):
                print(f"   ⚠️  Split '{split}' not found, skip.")
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

    # ------------------------------------------------------------------
    # Fingerprint extraction
    # ------------------------------------------------------------------

    def _extract_fingerprints(self, image, category, concept_id):
        msgs = _build_extraction_messages(image, category, concept_id)
        output = self.reasoner.model_interface.chat(msgs)
        # chat() returns (dict, str) from Qwen3VLModel or dict from ModelInterface
        if isinstance(output, tuple):
            _, raw_text = output
        else:
            raw_text = output.get("sequences", "")
        fingerprints = _parse_json_response(raw_text)
        return fingerprints

    # ------------------------------------------------------------------
    # Process single concept
    # ------------------------------------------------------------------

    def _process_concept(self, concept_data: dict) -> tuple[str, dict]:
        category   = concept_data["category"]
        concept_id = concept_data["concept_id"]
        images     = concept_data["images"]

        # Select representative image and top_k
        if self.use_clip_sel and self.clip_selector is not None:
            representative, top_k_paths = self.clip_selector.select(images, seed=self.seed)
        else:
            representative = images[0]
            top_k_paths = images[:3] # Fallback

        image = Image.open(representative).convert("RGB")
        fingerprints = self._extract_fingerprints(image, category, concept_id)

        concept_key = f"<{concept_id}>"
        entry = {
            "name":     concept_id,
            "image":    images,          
            "representative_image": representative,
            "top_k_images": top_k_paths, 
            "info":     fingerprints,    
            "category": category,
        }

        return concept_key, entry
    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Executes the complete pipeline for building the database.

        Returns:
            dict with success_count, total_concepts, database_path
        """
        print("\n" + "="*70)
        print("BUILD DATABASE — R2P-GEN FLUX Edition")
        print("="*70)
        Config.print_summary()
        print(f"  perva-data   : {self.perva_data_dir}")
        print(f"  split        : {self.dataset_split}")
        print(f"  CLIP select  : {self.use_clip_sel}")
        print(f"  debug        : {self.debug_mode} (limit={self.debug_limit})")
        print(f"  output       : {self.output_path}")
        print("="*70 + "\n")

        # 1. Scopri concetti
        print("[1/3] Discovering concepts...")
        all_concepts = self._get_concepts()
        if not all_concepts:
            print("❌ No concepts found. Check R2P_PERVA_DATA.")
            return {"success_count": 0, "total_concepts": 0}

        print(f"   Found {len(all_concepts)} concepts.")

        if self.debug_mode:
            all_concepts = all_concepts[:self.debug_limit]
            print(f"   DEBUG MODE: processing only {len(all_concepts)} concepts.")

        # 2. Estrai fingerprints
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
                print(f"\n   ⚠️  Error on '{cid}': {e}")
                continue

        # 3. Save
        print(f"\n[3/3] Saving database → {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self._database, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Done! {success}/{len(all_concepts)} concepts processati.")
        print(f"   Database: {os.path.abspath(self.output_path)}")

        # Cleanup modelli
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build R2P-GEN fingerprint database")
    parser.add_argument("--perva-data",  type=str, default=None,
                        help="Path a perva-data (default: R2P_PERVA_DATA env var)")
    parser.add_argument("--split",       type=str, default=None,
                        choices=["train", "test", "all"],
                        help="Dataset split (default: Config.BuildDatabase.DATASET_SPLIT)")
    parser.add_argument("--debug",       action="store_true", default=None,
                        help="Debug mode: process only the first N concepts")
    parser.add_argument("--debug-limit", type=int, default=None,
                        help="Number of concepts in debug mode (default: Config.BuildDatabase.DEBUG_LIMIT)")
    parser.add_argument("--no-clip",     action="store_true",
                        help="Disable CLIP selection (use first image)")
    args = parser.parse_args()

    builder = DatabaseBuilder(
        perva_data_dir    = args.perva_data or PERVA_DATA_DIR,
        dataset_split     = args.split,
        debug_mode        = args.debug,
        debug_limit       = args.debug_limit,
        use_clip_selection= not args.no_clip,
    )
    builder.build()