"""
pipeline/build_dataset.py

Costruisce il database di fingerprints per la pipeline R2P-GEN (FLUX Edition).

Workflow:
  1. Scopre tutti i concetti in perva-data (train/categoria/concept_id/)
  2. Seleziona l'immagine rappresentativa via CLIP centroid
  3. Estrae i fingerprints con Qwen3-VL (JSON strutturato)
  4. Salva database.json con concept_dict + path_to_concept

FIX 8: l'immagine rappresentativa selezionata via CLIP centroid viene ora
salvata esplicitamente in entry["representative_image"], cosi' generate.py
e flux_loop.py possono usarla invece di ricadere sempre su images[0].

Il flux_prompt NON viene generato qui: è costruito on-the-fly in flux_loop.py
tramite build_flux_prompt(fingerprints, target_context), che è deterministica
e dipende dal contesto target (può variare tra run diversi).

Variabili d'ambiente:
  R2P_PERVA_DATA   → path della cartella perva-data
                     default: /leonardo_work/IscrC_MUSE/tballari/perva-data
  R2P_MODELS_BASE  → base path modelli HuggingFace
                     default: usa repo-id (download automatico)
  R2P_CLUSTER_MODE → "true" per attivare cluster mode in config.py
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
# Costanti / env
# ---------------------------------------------------------------------------

# Path perva-data: env var > fallback cluster Leonardo
_DEFAULT_PERVA = "/leonardo_work/IscrC_MUSE/tballari/perva-data"
PERVA_DATA_DIR = os.environ.get("R2P_PERVA_DATA", _DEFAULT_PERVA)


# ---------------------------------------------------------------------------
# Prompt per Qwen3-VL (adattato da get_detailed_input_msgs_household di R2P)
# ---------------------------------------------------------------------------

# Esempio one-shot: stesso oggetto usato nel repo originale R2P (wnr = piatto ceramica).
# Se l'immagine non è disponibile, il sistema cade in zero-shot automaticamente.
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
    Costruisce i messaggi per Qwen3-VL per estrarre fingerprints in JSON.

    Usa one-shot se l'immagine esempio è disponibile, zero-shot altrimenti.

    Args:
        image:      PIL Image dell'oggetto da analizzare
        category:   categoria rilevata via CLIP (es. "bag")
        concept_id: identificatore unico del concetto (es. "alx")

    Returns:
        Lista messaggi nel formato [{"role": ..., "content": [...]}]
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
    Pulisce e parsa la risposta JSON di Qwen3-VL.
    Gestisce markdown fences e testo extra attorno al JSON.
    """
    # Rimuovi fences markdown
    cleaned = raw.strip().strip("```json").strip("```").strip()

    # Estrai solo il blocco JSON se c'è testo extra
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    else:
        raise ValueError(f"Nessun JSON trovato nella risposta: {raw[:200]}")

    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# CLIP image selector (mantenuto da build_database originale)
# ---------------------------------------------------------------------------

class _CLIPSelector:
    """Seleziona l'immagine più rappresentativa di un concept via CLIP centroid."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        print("   📎 Loading CLIP for image selection...")
        self.model = CLIPModel.from_pretrained(Config.Models.CLIP_MODEL).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(Config.Models.CLIP_MODEL)

    @torch.no_grad()
    def select(self, image_paths: list[str], seed: int = 42) -> str:
        """
        Restituisce il path dell'immagine più vicina al centroide CLIP.

        Args:
            image_paths: lista di path immagini del concept
            seed:        seed per shuffle deterministico tra le top-N (compatibilità R2P)

        Returns:
            Path dell'immagine selezionata
        """
        if len(image_paths) == 1:
            return image_paths[0]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.get_image_features(**inputs)
        # get_image_features può restituire un tensore o un oggetto — gestiamo entrambi
        if hasattr(output, 'image_embeds'):
            features = output.image_embeds
        elif hasattr(output, 'pooler_output'):
            features = output.pooler_output
        else:
            features = output  # già un tensore

        import torch.nn.functional as F
        features = F.normalize(features, p=2, dim=-1)

        centroid = features.mean(dim=0, keepdim=True)
        centroid = F.normalize(centroid, p=2, dim=-1)

        similarities = (features @ centroid.T).squeeze()
        best_idx = similarities.argmax().item()

        return image_paths[best_idx]

    def cleanup(self):
        del self.model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# DatabaseBuilder
# ---------------------------------------------------------------------------

class DatabaseBuilder:
    """
    Costruisce il database fingerprints per R2P-GEN.

    Per ogni concept:
      1. Seleziona immagine rappresentativa (CLIP centroid)
      2. Estrae fingerprints con Qwen3-VL
      3. Salva in database.json (incluso il path dell'immagine rappresentativa,
         FIX 8: campo "representative_image")

    Il flux_prompt viene costruito on-the-fly in flux_loop.py.
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
            out_name = "database.json"
        else:
            out_name = f"database_perva_{self.dataset_split}.json"
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
        """Scansiona perva-data e restituisce la lista di tutti i concept."""
        abs_path = os.path.abspath(self.perva_data_dir)
        print(f"   🔍 perva-data: {abs_path}")

        if not os.path.exists(abs_path):
            print(f"   ❌ Directory non trovata: {abs_path}")
            print(f"      Imposta R2P_PERVA_DATA per sovrascrivere il path.")
            return []

        splits = ["train", "test"] if self.dataset_split == "all" else [self.dataset_split]
        concepts = []

        for split in splits:
            split_dir = os.path.join(abs_path, split)
            if not os.path.exists(split_dir):
                print(f"   ⚠️  Split '{split}' non trovato, skip.")
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
        # I messaggi sono già in formato Qwen3-VL nativo —
        # li passiamo direttamente a model_interface.chat() senza adapter
        output = self.reasoner.model_interface.chat(msgs)
        # chat() restituisce (dict, str) da Qwen3VLModel oppure dict da ModelInterface
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
        """
        Processa un singolo concept:
          1. Seleziona immagine rappresentativa
          2. Estrae fingerprints
          3. Ritorna (concept_key, entry)

        FIX 8: l'immagine rappresentativa selezionata viene salvata in
        entry["representative_image"], cosi' generate.py e flux_loop.py
        possono usarla invece di ricadere sempre su images[0].
        """
        category   = concept_data["category"]
        concept_id = concept_data["concept_id"]
        images     = concept_data["images"]

        # Selezione immagine rappresentativa
        if self.use_clip_sel and self.clip_selector is not None:
            representative = self.clip_selector.select(images, seed=self.seed)
        else:
            representative = images[0]

        # Carica immagine
        image = Image.open(representative).convert("RGB")

        # Estrai fingerprints
        fingerprints = self._extract_fingerprints(image, category, concept_id)

        concept_key = f"<{concept_id}>"
        entry = {
            "name":     concept_id,
            "image":    images,          # tutte le immagini del concept
            # FIX 8: path dell'immagine scelta via CLIP centroid. Usata da
            # generate.py (immagine sorgente per Img2Img) e da flux_loop.py
            # (immagine di riferimento per verify/judge), con fallback a
            # images[0] se questo campo manca (database vecchi).
            "representative_image": representative,
            "info":     fingerprints,    # fingerprints JSON (niente sdxl_prompt)
            "category": category,
        }

        return concept_key, entry

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """
        Esegue la pipeline completa di build del database.

        Returns:
            dict con success_count, total_concepts, database_path
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
            print("❌ Nessun concept trovato. Controlla R2P_PERVA_DATA.")
            return {"success_count": 0, "total_concepts": 0}

        print(f"   Trovati {len(all_concepts)} concepts.")

        if self.debug_mode:
            all_concepts = all_concepts[:self.debug_limit]
            print(f"   DEBUG MODE: processing solo {len(all_concepts)} concepts.")

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
                print(f"\n   ⚠️  Errore su '{cid}': {e}")
                continue

        # 3. Salva
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
                        help="Debug mode: processa solo i primi N concepts")
    parser.add_argument("--debug-limit", type=int, default=None,
                        help="Numero concepts in debug mode (default: Config.BuildDatabase.DEBUG_LIMIT)")
    parser.add_argument("--no-clip",     action="store_true",
                        help="Disabilita CLIP selection (usa prima immagine)")
    args = parser.parse_args()

    builder = DatabaseBuilder(
        perva_data_dir    = args.perva_data or PERVA_DATA_DIR,
        dataset_split     = args.split,
        debug_mode        = args.debug,
        debug_limit       = args.debug_limit,
        use_clip_selection= not args.no_clip,
    )
    builder.build()