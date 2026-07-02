"""
pipeline/build_database_db.py (CENTROID VARIANT)

Costruisce il database dei concept estraendo i fingerprint JSON.
Usa una logica a Centroide (Image-to-Image) tramite CLIP per trovare
l'immagine più rappresentativa di un concept.
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
# Prompts for Qwen3-VL
# ---------------------------------------------------------------------------

_ONESHOT_IMAGE_PATH = os.path.join(project_root, "example_database", "wnr.jpg")

# ---------------------------------------------------------------------------
# Schemas for DreamBench (Living vs Inanimate)
# ---------------------------------------------------------------------------

_JSON_SCHEMA_OBJECT = {
    "general":          "a brief description of the object in one sentence.",
    "category":         "category of the object",
    "shape":            "shape of the object",
    "material":         "material of the object",
    "color":            "color of the object",
    "pattern":          "any distinct pattern if present, else 'none'",
    "brand/text":       "any text or brand visible on the object, else 'none'",
    "distinct features":"any distinct feature that makes the object unique"
}

_JSON_SCHEMA_LIVING = {
    "general":              "a brief description of the animal in one sentence.",
    "category":             "category or species (e.g., 'dog', 'cat')",
    "species_and_breed":    "the species and specific breed (e.g., 'Dog, Shiba Inu' or 'Cat, Tabby')",
    "coat_and_color":       "length, texture, and primary color of the fur/hair",
    "facial_features":      "distinctive traits of the face (e.g., eye color, ear shape, snout)",
    "distinctive_markings": "any unique spots, asymmetric color patches, or scars (e.g., 'white patch on the left paw')",
    "accessories":          "any collar, tag, harness, or clothing worn by the animal, else 'none'"
}

def _build_classification_message(image: Image.Image) -> list:
    """Step 1: Classifica se il soggetto è un essere vivente o un oggetto."""
    question = (
        "Look at the main entity in this image.\n"
        "Is it a living being (like an animal, dog, cat) or an inanimate object (like a backpack, can, toy)?\n"
        "Reply ONLY with the word 'LIVING' or 'OBJECT'."
    )
    return [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}
    ]

def _build_extraction_messages(
    image: Image.Image,
    category: str,
    subject_name: str,
    entity_type: str
) -> list:
    """Step 2: Estrae i fingerprint usando lo schema corretto isolando il soggetto."""
    
    if entity_type == "LIVING":
        schema = _JSON_SCHEMA_LIVING
        entity_desc = "living being (animal)"
    else:
        schema = _JSON_SCHEMA_OBJECT
        entity_desc = "inanimate object"

    question_test = (
        f"Describe the {entity_desc} in the image. The subject identifier is <{subject_name}>.\n"
        f"CRITICAL RULES:\n"
        f"1. Describe the entity as if it were an isolated 3D model in an empty white room.\n"
        f"2. DO NOT describe the background, the environment, the lighting, or the camera angle.\n"
        f"3. DO NOT describe its pose or current action (e.g., ignore if it is sitting, running, or held by someone).\n"
        f"4. Your response MUST be valid JSON and follow EXACTLY this format:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"The 'general' field MUST begin with '<{subject_name}> is ...'.\n"
        f"Respond ONLY with the JSON object. No extra text, no markdown fences."
    )

    msgs = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question_test}
        ]},
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
# CENTROID SELECTOR: CLIP Image-to-Image (Mean Embedding)
# ---------------------------------------------------------------------------

class _CentroidCLIPSelector:
    """
    Seleziona l'immagine più rappresentativa calcolando il centroide (media)
    degli embedding visivi di tutte le immagini e trovando quella più vicina.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        print("   📎 Loading CLIP for Centroid Image-to-Image selection...")
        self.model = CLIPModel.from_pretrained(Config.Models.CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(Config.Models.CLIP_MODEL)

    @torch.no_grad()
    def select(self, image_paths: list[str], k: int = 3) -> list[str]:
        """
        Ritorna le Top-K immagini più vicine al centroide del concept.
        """
        if len(image_paths) == 1:
            return [image_paths[0]]

        # 1. Carica solo le immagini (Niente testo)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        # 2. Estrai gli embedding visivi
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_image_features(**inputs)

        # get_image_features() può restituire un tensore o un oggetto
        # a seconda della versione di transformers installata
        if hasattr(outputs, 'image_embeds'):
            image_embeds = outputs.image_embeds
        elif hasattr(outputs, 'pooler_output'):
            image_embeds = outputs.pooler_output
        else:
            image_embeds = outputs  # già un tensore, versioni più vecchie
        
        # 3. Normalizza le feature individuali
        import torch.nn.functional as F
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # 4. Calcola il CENTROIDE (Media lungo la dimensione 0) e normalizzalo
        centroid = image_embeds.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=-1)

        # 5. Calcola la Cosine Similarity tra tutte le immagini e il centroide
        similarities = (image_embeds @ centroid.T).squeeze(-1)
        
        if similarities.dim() == 0:
            return [image_paths[0]]

        # 6. Seleziona le Top-K
        actual_k = min(k, len(image_paths))
        top_scores, top_indices = torch.topk(similarities, k=actual_k)

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
        perva_data_dir: str = None,
        dataset_split: str = None,
        debug_mode: bool = None,
        debug_limit: int = None,
        use_centroid_sel: bool = True,
        ignore_laion: bool = None,
        seed: int = None,
        device: str = None,
    ):
        self.perva_data_dir = perva_data_dir or os.environ.get("R2P_PERVA_DATA", "")
        self.dataset_split   = dataset_split   or Config.BuildDatabase.DATASET_SPLIT
        self.debug_mode      = debug_mode      if debug_mode is not None else Config.BuildDatabase.DEBUG_MODE
        self.debug_limit     = debug_limit     or Config.BuildDatabase.DEBUG_LIMIT
        self.use_centroid_sel= use_centroid_sel
        self.ignore_laion    = ignore_laion    if ignore_laion is not None else Config.BuildDatabase.IGNORE_LAION
        self.seed            = seed            or Config.BuildDatabase.SEED
        self.device          = device          or Config.GPU.DEVICE

        if Config.Database.CANONICAL_NAME:
            out_name = "database_db.json"
        else:
            out_name = f"database_perva_{self.dataset_split}.json"
        self.output_path = os.path.join(Config.Paths.DATABASE_DIR, out_name)

        self._reasoner      = None
        self._centroid_sel  = None

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
    def centroid_selector(self):
        if self._centroid_sel is None and self.use_centroid_sel:
            self._centroid_sel = _CentroidCLIPSelector(device=self.device)
        return self._centroid_sel

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
        # --- STEP 1: Classificazione Binaria ---
        class_msgs = _build_classification_message(image)
        class_output = self.reasoner.model_interface.chat(class_msgs)
        
        if isinstance(class_output, tuple):
            _, raw_class = class_output
        else:
            raw_class = class_output.get("sequences", "")
            
        is_living = "LIVING" in raw_class.upper()
        entity_type = "LIVING" if is_living else "OBJECT"

        # --- STEP 2: Estrazione JSON ---
        extract_msgs = _build_extraction_messages(image, category, category, entity_type)
        extract_output = self.reasoner.model_interface.chat(extract_msgs)
        
        if isinstance(extract_output, tuple):
            _, raw_json = extract_output
        else:
            raw_json = extract_output.get("sequences", "")
            
        fingerprints = _parse_json_response(raw_json)
        
        fingerprints["_entity_type"] = entity_type 
        return fingerprints

    def _process_concept(self, concept_data: dict) -> tuple[str, dict]:
        """
        Processa un singolo concept usando il Centroid Selector.
        """
        category   = concept_data["category"]
        concept_id = concept_data["concept_id"]
        images     = concept_data["images"]

        # 1. Selezione immagini Top-K guidata dal Centroide (Image-to-Image)
        if self.use_centroid_sel and self.centroid_selector is not None:
            # Nota: Non passiamo più la "category", non serve al centroide
            top_k_images = self.centroid_selector.select(images, k=3)
            representative = top_k_images[0]
        else:
            top_k_images = [images[0]]
            representative = images[0]

        image = Image.open(representative).convert("RGB")
        fingerprints = self._extract_fingerprints(image, category, concept_id)

        concept_key = f"<{category}>"
        entry = {
            "name":     category,
            "image":    images,
            "representative_image": representative,
            "top_k_images": top_k_images,
            "info":     fingerprints,
            "category": category,
        }

        return concept_key, entry

    def build(self) -> dict:
        print("\n" + "="*70)
        print("BUILD DATABASE — R2P-GEN FLUX Edition (CENTROID VARIANT)")
        print("="*70)
        Config.print_summary()
        
        print(f"  data-dir       : {self.perva_data_dir}")
        print(f"  split          : {self.dataset_split}")
        print(f"  CentroidCLIP   : {self.use_centroid_sel}")
        print(f"  debug          : {self.debug_mode} (limit={self.debug_limit})")
        print(f"  output         : {self.output_path}")
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
        if self._centroid_sel is not None:
            self._centroid_sel.cleanup()
            self._centroid_sel = None
        torch.cuda.empty_cache()

        return {
            "success_count":  success,
            "total_concepts": len(all_concepts),
            "database_path":  os.path.abspath(self.output_path),
        }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build R2P-GEN fingerprint database (Centroid Variant)")
    parser.add_argument("--data-dir",  type=str, default=None,
                        help="Path to perva-data (default: R2P_PERVA_DATA env var)")
    parser.add_argument("--split",       type=str, default=None,
                        choices=["train", "test", "all"],
                        help="Dataset split")
    parser.add_argument("--debug",       action="store_true", default=None,
                        help="Debug mode: processa solo i primi N concepts")
    parser.add_argument("--debug-limit", type=int, default=None,
                        help="Numero concepts in debug mode")
    parser.add_argument("--no-centroid", action="store_true",
                        help="Disabilita la logica a Centroide (prende semplicemente la prima immagine)")
    args = parser.parse_args()

    builder = DatabaseBuilder(
        perva_data_dir    = args.data_dir,
        dataset_split     = args.split,
        debug_mode        = args.debug,
        debug_limit       = args.debug_limit,
        use_centroid_sel  = not args.no_centroid,
    )
    builder.build()