"""
R2P-GEN Full Pipeline Orchestrator (FLUX Edition)

Questo modulo orchestra la pipeline R2P-GEN modulare:
- Stage 1: generate_only (Estrazione + FLUX Img2Img)
- Stage 2: verify_base (Verifica con Qwen3-VL + CLIP)
- Stage 3: recovery (Aggancio API per rigenerare i falliti)
- Stage 4: final_judge (Valutazione severa con InternVL2)
- Stage 5: full_auto (Esegue tutto in sequenza)
"""

import os
import json
import torch
import argparse

from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p
from pipeline.judge import FinalJudge
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning  # Step 1 output
from config import Config
import openai
from pipeline.recovery import vlm_rewrite_prompt, generate_recovery_http
from pipeline.prompts.flux_prompts import build_flux_prompt


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_first_image(content: dict) -> str | None:
    """
    Restituisce il path della prima immagine di un concept.
    Gestisce sia il campo 'image' (stringa) sia 'selected_images' (lista).
    Ritorna None se nessun path valido trovato — evita IndexError.
    """
    value = content.get("image") or content.get("selected_images")
    if not value:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return value  # già stringa


def _build_reasoner() -> Qwen3VLReasoning:
    """Istanzia Qwen3VLReasoning con i parametri da Config."""
    return Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,  # ← Qwen3-VL path
        device="cuda",
        torch_dtype=torch.bfloat16,            # Qwen3-VL è addestrato in bfloat16, non float16
        attn_implementation="flash_attention_2", # Qwen3 supporta FA2, più efficiente di sdpa
        seed=Config.Generate.SEED,
    )


# ---------------------------------------------------------------------------
# Stage 1 — Generate Only
# ---------------------------------------------------------------------------

def stage_generate_only(
    database_path: str,
    output_dir: str,
    num_shards: int = 1,
    shard_index: int = 0,
) -> dict:
    """Fase 1: Estrazione + FLUX Img2Img (con sharding opzionale)."""
    print(f"\n{'='*70}\n🚀 STAGE: GENERATE ONLY "
          f"[shard {shard_index}/{num_shards}]\n{'='*70}")
    ensure_output_dir(output_dir)

    generator = Generator(
        database_path=database_path,
        output_dir=output_dir,
        num_shards=num_shards,      # Generator deve accettare questi kwargs;
        shard_index=shard_index,    # se non li accetta ancora, aggiungili lì.
    )
    stats = generator.generate_all()
    generator.cleanup()
    cleanup_gpu()

    print(f"\n📊 Generazione: {stats['success']} OK, {stats['failed']} falliti.")
    return stats


# ---------------------------------------------------------------------------
# Stage 2 — Verify Base
# ---------------------------------------------------------------------------

def stage_verify_base(database_path: str, output_dir: str) -> str:
    """Fase 2: Verifica con Qwen3-VL + CLIP. Salva rejected_concepts.json."""
    print(f"\n{'='*70}\n📍 STAGE: VERIFY BASE (Qwen3-VL)\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print("   Caricamento modelli di verifica...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    verified_count = 0
    rejected_dict = {}

    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")

        if not os.path.exists(gen_image_path):
            print(f"   ⚠️  {concept_id}: immagine non trovata → fallita.")
            rejected_dict[concept_id] = {"reason": "Image not found"}
            continue

        ref_image_path = _get_first_image(content)
        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: nessuna immagine di riferimento → skip.")
            rejected_dict[concept_id] = {"reason": "No reference image"}
            continue

        fingerprints = content.get("info", {})

        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=gen_image_path,
            ref_image_path=ref_image_path,
            fingerprints=fingerprints,
        )

        if verification["is_verified"]:
            verified_count += 1
            print(f"   ✅ {concept_id}: PASS ({verification['score']:.2f})")
        else:
            print(f"   ❌ {concept_id}: FAIL ({verification['score']:.2f})")
            rejected_dict[concept_id] = {
                "score": verification["score"],
                "error_type": "attribute",
                "missing_details": verification.get("failed_attributes", []),
                "details": verification,
            }

    del reasoner
    del clip_calculator
    cleanup_gpu()

    rejected_path = os.path.join(output_dir, "rejected_concepts.json")
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected_dict, f, indent=4)

    total = len(concept_dict)
    print(f"\n📊 Verifica: {verified_count}/{total} passed. "
          f"{len(rejected_dict)} rejected → {rejected_path}")
    return rejected_path


# ---------------------------------------------------------------------------
# Stage 3 — Recovery
# ---------------------------------------------------------------------------

def stage_recovery(
    database_path: str,
    rejected_path: str,
    output_dir: str,
) -> None:
    """Fase 3: Recovery tramite API (closed-loop ospedaliero)."""
    print(f"\n{'='*70}\n🚑 STAGE: RECOVERY API\n{'='*70}")

    if not os.path.exists(rejected_path):
        print("   ✅ Nessun file rejected trovato. Niente da recuperare!")
        return

    with open(rejected_path, "r", encoding="utf-8") as f:
        rejected_dict = json.load(f)

    if not rejected_dict:
        print("   ✅ Lista rejected vuota. Niente da recuperare!")
        return

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print(f"   Trovati {len(rejected_dict)} concetti da curare.")

    # Config API — centralizzati qui, spostare in Config se necessario
    VLM_URL        = getattr(Config.Models, "RECOVERY_VLM_URL",  "http://localhost:8000/v1")
    VLM_MODEL_NAME = getattr(Config.Models, "RECOVERY_VLM_NAME", Config.Models.VLM_MODEL)
    FLUX_URL       = getattr(Config.Models, "RECOVERY_FLUX_URL", "http://localhost:8766")
    MAX_ATTEMPTS   = Config.Refine.MAX_ITERATIONS

    try:
        vlm_client = openai.OpenAI(api_key="EMPTY", base_url=VLM_URL)
    except Exception as e:
        print(f"   ❌ Errore inizializzazione client VLM: {e}")
        return

    print("   Caricamento modelli di verifica per il loop di recovery...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    recovered_count = 0
    graveyard_count = 0

    for concept_id, fail_data in rejected_dict.items():
        print(f"\n   🚑 Paziente: {concept_id} | "
              f"Problema: {fail_data.get('missing_details', ['Unknown'])}")

        content = concept_dict.get(concept_id, {})

        source_image_path = _get_first_image(content)
        if source_image_path is None:
            print(f"      ⚠️  Nessuna immagine sorgente per {concept_id} → graveyard.")
            graveyard_count += 1
            continue

        fingerprints    = content.get("info", {})
        target_context  = Config.get_background_template()
        current_prompt  = build_flux_prompt(fingerprints, target_context)
        missing_details = fail_data.get("missing_details", [])
        gen_image_path  = os.path.join(output_dir, f"{concept_id}_generated.png")
        is_fixed        = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"      🔄 Tentativo {attempt}/{MAX_ATTEMPTS}...")

            # 1. Prescrizione
            new_prompt = vlm_rewrite_prompt(
                vlm_client=vlm_client,
                model_name=VLM_MODEL_NAME,
                original_prompt=current_prompt,
                missing_details=missing_details,
                attempt=attempt,
            )
            print(f"         📝 Prompt: {new_prompt[:80]}...")

            # 2. Terapia
            new_seed = Config.Generate.SEED + attempt
            success = generate_recovery_http(
                flux_url=FLUX_URL,
                source_image_path=source_image_path,
                prompt=new_prompt,
                seed=new_seed,
                output_path=gen_image_path,
            )

            if not success:
                print("         ⚠️ Rigenerazione API fallita, prossimo tentativo.")
                continue

            # 3. Visita di controllo
            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=gen_image_path,
                ref_image_path=source_image_path,
                fingerprints=fingerprints,
            )

            if verification["is_verified"]:
                print(f"         ✅ GUARITO al tentativo {attempt}.")
                is_fixed = True
                recovered_count += 1
                break
            else:
                missing_details = verification.get("failed_attributes", missing_details)
                current_prompt  = new_prompt
                print(f"         ❌ Permangono problemi: {missing_details}")

        if not is_fixed:
            print(f"      💀 UNRECOVERABLE: {concept_id} → graveyard.")
            graveyard_count += 1

    print(f"\n📊 Recovery: {recovered_count} salvati | {graveyard_count} graveyard")

    del reasoner
    del clip_calculator
    cleanup_gpu()


# ---------------------------------------------------------------------------
# Stage 4 — Final Judge
# ---------------------------------------------------------------------------

def stage_final_judge(database_path: str, output_dir: str) -> None:
    """Fase 4: Giudizio finale indipendente (InternVL2)."""
    print(f"\n{'='*70}\n⚖️  STAGE: FINAL JUDGE\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    judge = FinalJudge(
        threshold=Config.Refine.TARGET_ACCURACY,
        use_dino=True,
        use_clip=True,
        use_vqa=True,
    )

    concept_dict = database.get("concept_dict", {})
    results = {}

    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
        if not os.path.exists(gen_image_path):
            continue

        ref_image_path = _get_first_image(content)
        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: nessuna reference → skip.")
            continue

        fingerprints = content.get("info", {})

        try:
            judge_eval = judge.evaluate(
                generated_image=gen_image_path,
                reference_image=ref_image_path,
                fingerprints=fingerprints,
                prompt=fingerprints.get("flux_prompt",
                       fingerprints.get("sdxl_prompt", "")),
            )
            results[concept_id] = judge_eval.to_dict()
            status = "✅" if judge_eval.passed else "❌"
            print(f"   {status} {concept_id} | "
                  f"TIFA: {judge_eval.tifa_score:.1%} | "
                  f"DINO: {judge_eval.dino_i:.3f}")
        except Exception as e:
            print(f"   ⚠️ Errore su {concept_id}: {e}")

    judge.cleanup()

    results_path = os.path.join(output_dir, "final_judge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n📁 Risultati finali → {results_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2P-GEN Modulare (FLUX Edition)")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output",   type=str, default="output")
    parser.add_argument("--stage",
                        choices=["generate_only", "verify_base",
                                 "recovery", "final_judge", "full_auto"],
                        required=True)
    parser.add_argument("--num-shards",  type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    rejected_path = os.path.join(args.output, "rejected_concepts.json")

    if args.stage == "generate_only":
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index)

    elif args.stage == "verify_base":
        stage_verify_base(args.database, args.output)

    elif args.stage == "recovery":
        stage_recovery(args.database, rejected_path, args.output)

    elif args.stage == "final_judge":
        stage_final_judge(args.database, args.output)

    elif args.stage == "full_auto":
        print("\n🚀 AVVIO PIPELINE FULL AUTO")
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index)
        rejected_path = stage_verify_base(args.database, args.output)
        stage_recovery(args.database, rejected_path, args.output)
        stage_final_judge(args.database, args.output)
        print("\n🏁 PIPELINE COMPLETATA")