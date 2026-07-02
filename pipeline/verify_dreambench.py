"""
pipeline/verify_dreambench.py

Verify per il benchmark DreamBench. Script SEPARATO da
pipeline_orchestrator.stage_verify_base perché la struttura di output è
diversa (concept x prompt x immagine, non un singolo file per concept) e
perché qui si applica una regola di esclusione specifica per i 5 prompt di
property modification (vedi dreambench_prompts.PROPERTY_MODIFICATION_INDICES).

Regola di esclusione (decisione concordata):
- Per i 5 prompt "a red {0}", "a purple {0}", "a shiny {0}", "a wet {0}",
  "a cube shaped {0}" il prompt stesso richiede di cambiare colore/materiale/
  forma del soggetto -> il verify per attributi (che controlla la fedeltà di
  quegli stessi campi rispetto al reference) andrebbe in contraddizione
  logica. Queste immagini vengono quindi SKIPPATE dal verify e passano
  dirette alla Fase 4 (metriche ufficiali).
- Per gli altri 20 prompt il fingerprint va verificato COMPLETO, senza
  esclusioni per-campo.

Output: un unico file rejected_dreambench.json con chiave composita
"<concept_id>/{prompt_idx:02d}/{img_idx}" per ogni immagine fallita.
"""

import os
import json
import argparse
from tqdm import tqdm

from pipeline.verify import verify_generation_r2p
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.prompts.dreambench_prompts import (
    get_prompts_for_entity_type,
    is_property_modification_prompt,
)
from config import Config


def _get_first_image(content: dict) -> str | None:
    if content.get("representative_image"):
        return content["representative_image"]
    top_k = content.get("top_k_images")
    if top_k:
        return top_k[0]
    value = content.get("image")
    if isinstance(value, list) and value:
        return value[0]
    return None


def _sanitize_folder_name(name: str) -> str:
    return name.strip("<>")


def _build_reasoner():
    import torch
    from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
    return Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        seed=Config.Generate.SEED,
    )


def run_dreambench_verify(database_path: str, output_dir: str) -> str:
    print(f"\n{'='*70}")
    print("📍 VERIFY DREAMBENCH (Qwen3-VL + CLIP) — esclude i 5 prompt property-mod")
    print(f"{'='*70}\n")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print("   Caricamento modelli di verifica...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    rejected_dict = {}
    verified_count = 0
    skipped_count = 0
    total_checked = 0

    for concept_id, content in tqdm(concept_dict.items(), desc="Concepts"):
        fingerprints = content.get("info", {})
        entity_type = fingerprints.get("_entity_type", "OBJECT")
        ref_image_path = _get_first_image(content)
        clean_name = _sanitize_folder_name(concept_id)

        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: nessuna immagine di riferimento → skip totale.")
            continue

        prompts = get_prompts_for_entity_type(entity_type)
        concept_dir = os.path.join(output_dir, clean_name)
        if not os.path.isdir(concept_dir):
            continue

        for prompt_idx in range(len(prompts)):
            prompt_dir = os.path.join(concept_dir, f"{prompt_idx:02d}")
            if not os.path.isdir(prompt_dir):
                continue

            # --- Regola di esclusione: property modification ---
            if is_property_modification_prompt(prompt_idx):
                skipped_count += len([
                    f for f in os.listdir(prompt_dir) if f.endswith(".png")
                ])
                continue

            for fname in sorted(os.listdir(prompt_dir)):
                if not fname.endswith(".png"):
                    continue
                img_idx = os.path.splitext(fname)[0]
                gen_image_path = os.path.join(prompt_dir, fname)
                composite_key = f"{concept_id}/{prompt_idx:02d}/{img_idx}"
                total_checked += 1

                verification = verify_generation_r2p(
                    reasoner=reasoner,
                    clip_calculator=clip_calculator,
                    gen_image_path=gen_image_path,
                    ref_image_path=ref_image_path,
                    fingerprints=fingerprints,
                )

                if verification["is_verified"]:
                    verified_count += 1
                else:
                    rejected_dict[composite_key] = {
                        "concept_id":      concept_id,
                        "prompt_idx":       prompt_idx,
                        "img_idx":          img_idx,
                        "score":            verification["score"],
                        "error_type":       "attribute",
                        "missing_details":  verification.get("failed_attributes", []),
                        "gen_image_path":   gen_image_path,
                        "ref_image_path":   ref_image_path,
                        "details":          verification,
                    }

    del reasoner
    del clip_calculator
    import torch
    torch.cuda.empty_cache()

    rejected_path = os.path.join(output_dir, "rejected_dreambench.json")
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected_dict, f, indent=4, ensure_ascii=False)

    print(f"\n📊 Verify completato:")
    print(f"   Controllate (20 prompt validi): {total_checked}")
    print(f"   ✅ Passate:  {verified_count}")
    print(f"   ❌ Rejected: {len(rejected_dict)}")
    print(f"   ⏭️  Skippate (5 prompt property-mod, no verify): {skipped_count}")
    print(f"   📁 Report → {rejected_path}")

    return rejected_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify DreamBench (R2P-GEN)")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output",   type=str, required=True,
                        help="Cartella output_dreambench con le immagini generate")
    args = parser.parse_args()

    run_dreambench_verify(args.database, args.output)