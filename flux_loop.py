"""
R2P-GEN Full Pipeline Orchestrator (FLUX Edition)

This module orchestrates the R2P-GEN modular pipeline:
- Stage 1: generate_only (Extraction + FLUX Img2Img)
- Stage 2: verify_base (Verification with Qwen3-VL + CLIP)
- Stage 3: refine (API integration for regenerating failed cases)
- Stage 4: final_judge (Severe evaluation with InternVL2)
- Stage 5: full_auto (Executes everything in sequence)
"""

import os
import json
import torch
import argparse
import numpy as np

from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p, _extract_attributes_for_clip
from pipeline.judge import FinalJudge
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
from config import Config
import openai
from pipeline.refine import qwen3_rewrite_prompt, generate_recovery_http, _generate_batch_http
from pipeline.prompts.flux_prompts import build_flux_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_first_image(content: dict) -> str | None:
    if content.get("representative_image"):
        return content["representative_image"]
    top_k = content.get("top_k_images")
    if top_k:
        return top_k[0]
    value = content.get("image") or content.get("selected_images")
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get_best_reference_for_recovery(
    content: dict,
    missing_details: list,
    clip_calculator,
) -> str:
    from PIL import Image
    candidates = content.get("top_k_images") or []
    if not candidates:
        return _get_first_image(content)
    if not missing_details:
        return candidates[0]
    best_path, best_score = candidates[0], -1.0
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            img = Image.open(path).convert("RGB")
            score, _ = clip_calculator.compute_attribute_score(img, missing_details)
            if score > best_score:
                best_score = score
                best_path = path
        except Exception:
            continue
    return best_path


def _build_reasoner() -> Qwen3VLReasoning:
    return Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        seed=Config.Generate.SEED,
    )


# ---------------------------------------------------------------------------
# Per-attempt log helpers
# ---------------------------------------------------------------------------

def _build_attempt_log_entry(
    attempt: int,
    prompt_used: str,
    verification: dict,
    image_path: str,
) -> dict:
    """
    Builds a detailed per-attempt log entry mirroring the diagnostic
    precision of verify output: per-attribute VLM and CLIP scores,
    not just averages.
    """
    # Per-attribute VLM scores (Phase 1 single_check entries)
    vlm_per_attribute = {}
    for item in verification.get("vlm_history", []):
        if item.get("phase") == "single_check":
            vlm_per_attribute[item["attribute"]] = {
                "score":    item["score"],
                "yes_conf": item["yes_conf"],
                "no_conf":  item["no_conf"],
                "response": item.get("response", ""),
            }

    # Per-attribute CLIP scores (gen vs ref delta)
    clip_gen = verification.get("clip_details", {}).get("gen", {})
    clip_ref = verification.get("clip_details", {}).get("ref", {})
    all_attrs = set(list(clip_gen.keys()) + list(clip_ref.keys()))
    clip_per_attribute = {
        attr: {
            "gen_score": clip_gen.get(attr, 0.0),
            "ref_score": clip_ref.get(attr, 0.0),
            "delta":     clip_gen.get(attr, 0.0) - clip_ref.get(attr, 0.0),
        }
        for attr in all_attrs
    }

    return {
        "attempt":            attempt,
        "prompt_used":        prompt_used,
        "score":              verification.get("score", 0.0),
        "method":             verification.get("method", ""),
        "success":            verification["is_verified"],
        "missing_attributes": verification.get("failed_attributes", []),
        "image_path":         image_path,
        "vlm_per_attribute":  vlm_per_attribute,
        "clip_per_attribute": clip_per_attribute,
    }


def _build_failed_generation_log_entry(
    attempt: int,
    prompt_used: str,
    current_missing: list,
) -> dict:
    """Log entry for a batch item where FLUX generation itself failed."""
    return {
        "attempt":            attempt,
        "prompt_used":        prompt_used,
        "score":              None,
        "method":             "flux_generation_failed",
        "success":            False,
        "missing_attributes": current_missing,
        "image_path":         None,
        "vlm_per_attribute":  {},
        "clip_per_attribute": {},
    }


# ---------------------------------------------------------------------------
# Stage 1 — Generate Only
# ---------------------------------------------------------------------------

# def stage_generate_only(
#     database_path: str,
#     output_dir: str,
#     num_shards: int = 1,
#     shard_index: int = 0,
# ) -> dict:
#     print(f"\n{'='*70}\n🚀 STAGE: GENERATE ONLY "
#           f"[shard {shard_index}/{num_shards}]\n{'='*70}")
#     ensure_output_dir(output_dir)

#     generator = Generator(
#         database_path=database_path,
#         output_dir=output_dir,
#         num_shards=num_shards,
#         shard_index=shard_index,
#     )
#     stats = generator.generate_all()

def stage_generate_only(
    database_path: str,
    output_dir: str,
    num_shards: int = 1,
    shard_index: int = 0,
    use_naive_prompt: bool = False,
    no_image_cond: bool = False,
) -> dict:
    print(f"\n{'='*70}\n🚀 STAGE: GENERATE ONLY "
          f"[shard {shard_index}/{num_shards}]\n{'='*70}")
    ensure_output_dir(output_dir)

    generator = Generator(
        database_path=database_path,
        output_dir=output_dir,
        num_shards=num_shards,
        shard_index=shard_index,
        use_naive_prompt=use_naive_prompt,
        no_image_cond=no_image_cond,
    )
    stats = generator.generate_all()
    generator.cleanup()
    cleanup_gpu()

    print(f"\n📊 Generation: {stats['success']} OK, {stats['failed']} failed.")
    return stats


# ---------------------------------------------------------------------------
# Stage 2 — Verify Base
# ---------------------------------------------------------------------------

def stage_verify_base(database_path: str, output_dir: str) -> str:
    print(f"\n{'='*70}\n📍 STAGE: VERIFY BASE (Qwen3-VL)\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print("   Loading verification models...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    verified_count = 0
    rejected_dict = {}

    for concept_id, content in concept_dict.items():
        gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")

        if not os.path.exists(gen_image_path):
            print(f"   ⚠️  {concept_id}: image not found → failed.")
            rejected_dict[concept_id] = {"reason": "Image not found"}
            continue

        ref_image_path = _get_first_image(content)
        if ref_image_path is None:
            print(f"   ⚠️  {concept_id}: no reference image → skip.")
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
                "score":           verification["score"],
                "error_type":      "attribute",
                "missing_details": verification.get("failed_attributes", []),
                "details":         verification,
            }

    del reasoner
    del clip_calculator
    cleanup_gpu()

    rejected_path = os.path.join(output_dir, "rejected_concepts.json")
    with open(rejected_path, "w", encoding="utf-8") as f:
        json.dump(rejected_dict, f, indent=4)

    total = len(concept_dict)
    print(f"\n📊 Verification: {verified_count}/{total} passed. "
          f"{len(rejected_dict)} rejected → {rejected_path}")
    return rejected_path


# ---------------------------------------------------------------------------
# Stage 3 — Refine (batch mode)
# ---------------------------------------------------------------------------

def stage_refine(
    database_path: str,
    rejected_path: str,
    output_dir: str,
) -> None:
    """Stage 3: Refine via API (closed-loop) in batch on FLUX."""
    print(f"\n{'='*70}\n🚑 STAGE: RECOVERY (BATCH MODE)\n{'='*70}")

    if not os.path.exists(rejected_path):
        print("   ✅ No rejected file found. Nothing to recover!")
        return

    if os.path.getsize(rejected_path) == 0:
        print("   ❌ rejected_concepts.json exists but is empty. Abort.")
        return

    with open(rejected_path, "r", encoding="utf-8") as f:
        try:
            rejected_dict = json.load(f)
        except json.JSONDecodeError as e:
            print(f"   ❌ rejected_concepts.json is not valid JSON: {e}. Abort.")
            return

    if not rejected_dict:
        print("   ✅ List rejected is empty. Nothing to recover!")
        return

    if not os.path.exists(database_path):
        print(f"   ❌ Database not found: {database_path}. Abort.")
        return

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    print(f"   Found {len(rejected_dict)} concepts to recover.")

    FLUX_URL     = getattr(Config.Models, "RECOVERY_FLUX_URL", "http://127.0.0.1:8766")
    MAX_ATTEMPTS = Config.Refine.MAX_ITERATIONS

    print("   Loading Qwen3-VL and CLIP in VRAM (GPU 0) for the recovery loop...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    recovered_count = 0
    graveyard_count = 0
    recovery_report = {}

    # -----------------------------------------------------------------------
    # 1. Initialize state for each concept
    # -----------------------------------------------------------------------
    concept_states = {}
    for concept_id, fail_data in rejected_dict.items():
        content = concept_dict.get(concept_id, {})
        source_image_path = _get_first_image(content)

        if source_image_path is None:
            print(f"      ⚠️  No source image found for {concept_id} → graveyard.")
            recovery_report[concept_id] = {
                "status":               "unrecoverable",
                "attempts":             0,
                "final_score":          fail_data.get("score", 0.0),
                "last_missing_details": fail_data.get("missing_details", []),
                "original_fail_reason": fail_data.get("missing_details", []),
                "recovered_image_path": None,
                "original_prompt":      None,
                "last_rewritten_prompt": None,
                "reason":               "No source image",
                "attempts_log":         [],
            }
            graveyard_count += 1
            continue

        fingerprints    = content.get("info", {})
        target_context  = Config.get_background_template()
        current_prompt  = build_flux_prompt(fingerprints, target_context)
        missing_details = fail_data.get("missing_details", [])

        concept_states[concept_id] = {
            "content":               content,
            "fingerprints":          fingerprints,
            "current_prompt":        current_prompt,
            "missing_details":       missing_details,
            "is_fixed":              False,
            "verification":          None,
            # _generated.png at the start of recovery is the original failed image.
            # It stays _generated.png until a recovery attempt succeeds, at which
            # point it gets renamed to _generated_rejected_attempt0.png and the
            # successful attempt overwrites _generated.png.
            "gen_image_path":        os.path.join(output_dir, f"{concept_id}_generated.png"),
            "original_prompt":       current_prompt,
            "last_rewritten_prompt": None,
            "attempts_history":      [],
            "attempts_log":          [],
            "original_fail_reason":  fail_data.get("missing_details", []),
        }

    # -----------------------------------------------------------------------
    # 2. Loop per tentative — batch over all the active concepts
    # -----------------------------------------------------------------------
    for attempt in range(1, MAX_ATTEMPTS + 1):
        active = {cid: s for cid, s in concept_states.items() if not s["is_fixed"]}
        if not active:
            break

        print(f"\n   {'='*50}")
        print(f"   🔄 TENTATIVE {attempt}/{MAX_ATTEMPTS} — {len(active)} active concepts")
        print(f"   {'='*50}")

        prompts_batch  = []
        seeds_batch    = []
        paths_batch    = []
        cids_batch     = []
        sources_batch  = []

        for concept_id, state in active.items():
            missing_details_before = list(state["missing_details"])

            best_source = _get_best_reference_for_recovery(
                state["content"], state["missing_details"], clip_calculator
            )

            # For attempt 1: failed_image_path = _generated.png (original failed)
            # For attempt N>1: failed_image_path = _generated_attemptN-1.png
            failed_image_path = (
                state["gen_image_path"]
                if attempt == 1
                else (state["attempts_log"][-1].get("image_path") if state["attempts_log"] else None)
            )

            new_prompt = qwen3_rewrite_prompt(
                reasoner=reasoner,
                original_prompt=state["current_prompt"],
                missing_details=state["missing_details"],
                attempt=attempt,
                failed_image_path=failed_image_path,
                attempts_history=state["attempts_history"],
                fingerprints=state["fingerprints"],  
            )
            print(f"      [{concept_id}] 📝 {new_prompt[:70]}...")

            attempt_image_path = os.path.join(
                output_dir, f"{concept_id}_generated_attempt{attempt}.png"
            )

            # Different SEED to avoid generating the same image again. Each attempt gets a new seed.
            new_seed = Config.Generate.SEED + attempt * 1000

            prompts_batch.append(new_prompt)
            seeds_batch.append(new_seed)
            paths_batch.append(attempt_image_path)
            cids_batch.append(concept_id)
            sources_batch.append(best_source)

            state["last_rewritten_prompt"]   = new_prompt
            state["_missing_details_before"] = missing_details_before
            state["_best_source"]            = best_source

        # ---- Batch FLUX ----
        print(f"\n      🚀 Sending FLUX batch: {len(prompts_batch)} images...")
        batch_results = _generate_batch_http(
            flux_url=FLUX_URL,
            source_image_paths=sources_batch,
            prompts=prompts_batch,
            seeds=seeds_batch,
            output_paths=paths_batch,
        )

        # ---- Verification for each concept ----
        for i, concept_id in enumerate(cids_batch):
            state                  = concept_states[concept_id]
            attempt_image_path     = paths_batch[i]
            success                = batch_results[i]
            new_prompt             = state["last_rewritten_prompt"]
            missing_details_before = state["_missing_details_before"]

            if not success:
                print(f"      [{concept_id}] ⚠️ FLUX generation failed.")
                state["attempts_log"].append(
                    _build_failed_generation_log_entry(
                        attempt=attempt,
                        prompt_used=new_prompt,
                        current_missing=state["missing_details"],
                    )
                )
                continue

            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=attempt_image_path,
                ref_image_path=state["_best_source"],
                fingerprints=state["fingerprints"],
            )

            # Per-attribute detailed log entry
            state["attempts_log"].append(
                _build_attempt_log_entry(
                    attempt=attempt,
                    prompt_used=new_prompt,
                    verification=verification,
                    image_path=attempt_image_path,
                )
            )

            state["attempts_history"].append({
                "attempt":      attempt,
                "missing_before": missing_details_before,
                "missing_after":  verification.get("failed_attributes", state["missing_details"]),
                "improved":       len(verification.get("failed_attributes", state["missing_details"]))
                                  < len(missing_details_before),
            })

            state["verification"] = verification

            if verification["is_verified"]:
                print(f"      [{concept_id}] ✅ Saved at attempt {attempt}!")
                import shutil

                # Rename the previous _generated.png to _generated_rejected_attempt0.png
                # (attempt 1 success → rejected = attempt0 = original generated image)
                previous_attempt_num = attempt - 1
                rejected_name = os.path.join(
                    output_dir,
                    f"{concept_id}_generated_rejected_attempt{previous_attempt_num}.png",
                )
                if os.path.exists(state["gen_image_path"]):
                    os.rename(state["gen_image_path"], rejected_name)
                    print(f"         📦 Previous image saved as: {os.path.basename(rejected_name)}")

                shutil.copy2(attempt_image_path, state["gen_image_path"])
                state["is_fixed"] = True
                recovered_count += 1

            else:
                new_missing = verification.get("failed_attributes", state["missing_details"])

                # Filter out hallucinated long strings
                MAX_ATTR_LENGTH = 80
                new_missing = [a for a in new_missing if len(a) <= MAX_ATTR_LENGTH]
                if not new_missing:
                    new_missing = [
                        a for a in state["original_fail_reason"]
                        if len(a) <= MAX_ATTR_LENGTH
                    ][:3]

                state["missing_details"] = new_missing
                state["current_prompt"]  = new_prompt
                print(f"      [{concept_id}] ❌ Remaining: {state['missing_details']}")

    # -----------------------------------------------------------------------
    # 3. Compilation of final report
    # -----------------------------------------------------------------------
    for concept_id, state in concept_states.items():
        is_fixed     = state["is_fixed"]
        verification = state["verification"]

        if is_fixed:
            attempt_success = None
            for log in reversed(state["attempts_log"]):
                if log.get("success"):
                    attempt_success = log["attempt"]
                    break
            recovery_report[concept_id] = {
                "status":               "recovered",
                "attempts":             attempt_success or 0,
                "final_score":          verification.get("score", 0.0) if verification else 0.0,
                "last_missing_details": [],
                "recovered_image_path": state["gen_image_path"],
                "original_prompt":      state["original_prompt"],
                "last_rewritten_prompt": state["last_rewritten_prompt"],
                "original_fail_reason": state["original_fail_reason"],
                "reason":               "",
                "attempts_log":         state["attempts_log"],
            }
        else:
            graveyard_count += 1
            recovery_report[concept_id] = {
                "status":               "unrecoverable",
                "attempts":             MAX_ATTEMPTS,
                "final_score":          verification.get("score", 0.0) if verification else 0.0,
                "last_missing_details": state["missing_details"],
                "recovered_image_path": None,
                "original_prompt":      state["original_prompt"],
                "last_rewritten_prompt": state["last_rewritten_prompt"],
                "original_fail_reason": state["original_fail_reason"],
                "reason":               "Max attempts reached",
                "attempts_log":         state["attempts_log"],
            }

    report_path = os.path.join(output_dir, "recovery_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(recovery_report, f, indent=4)
    print(f"\n📋 Recovery report saved → {report_path}")
    print(f"\n📊 Recovery: {recovered_count} saved | {graveyard_count} graveyard")

    del reasoner
    del clip_calculator
    cleanup_gpu()

# ---------------------------------------------------------------------------
# Stage 4 — Final Judge
# ---------------------------------------------------------------------------

def stage_final_judge(database_path: str, output_dir: str) -> None:
    print(f"\n{'='*70}\n⚖️  STAGE: FINAL JUDGE\n{'='*70}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)

    prompts_path = os.path.join(output_dir, "prompts.json")
    generated_prompts = {}
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            generated_prompts = json.load(f)
        print(f"   📖 File prompts.json loaded.")

    recovery_path = os.path.join(output_dir, "recovery_results.json")
    recovery_data = {}
    if os.path.exists(recovery_path):
        with open(recovery_path, "r", encoding="utf-8") as f:
            recovery_data = json.load(f)
        print(f"   🚑 File recovery_results.json loaded.")

    judge = FinalJudge(
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
            print(f"   ⚠️  {concept_id}: no reference found → skip.")
            continue

        fingerprints = content.get("info", {})

        # --- Prompt decision ---
        actual_prompt = ""
        if concept_id in generated_prompts:
            actual_prompt = generated_prompts[concept_id].get("flux_prompt", "")
        
        # If the concept was recovered, use the last rewritten prompt for evaluation
        if concept_id in recovery_data:
            rec_status = recovery_data[concept_id].get("status")
            rec_prompt = recovery_data[concept_id].get("last_rewritten_prompt")
            if rec_status == "recovered" and rec_prompt:
                actual_prompt = rec_prompt
                print(f"   🔄 Using rewritten prompt for {concept_id}")

        try:
            judge_eval = judge.evaluate(
                generated_image=gen_image_path,
                reference_image=ref_image_path,
                fingerprints=fingerprints,
                prompt=actual_prompt, 
            )
            results[concept_id] = judge_eval.to_dict()
            print(f"   ✅ {concept_id} | "
                  f"CLIP-I: {judge_eval.clip_i:.3f} | "
                  f"DINO-I: {judge_eval.dino_i:.3f} | "
                  f"CLIP-T: {judge_eval.clip_t:.3f} | "
                  f"TIFA: {judge_eval.tifa_score:.1%}")
        except Exception as e:
            print(f"   ⚠️ Error on {concept_id}: {e}")

    judge.cleanup()

    results_path = os.path.join(output_dir, "final_judge_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\n📁 Final results → {results_path}")
    
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2P-GEN Modular (FLUX Edition)")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--output",   type=str, default="output")
    parser.add_argument("--stage",
                        choices=["generate_only", "verify_base",
                                 "refine", "final_judge", "full_auto"],
                        required=True)
    parser.add_argument("--num-shards",  type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    
    # ablation's flags
    parser.add_argument("--naive-prompt", action="store_true", help="Use base prompt without fingerprints")
    parser.add_argument("--no-image-cond", action="store_true", help="Disable img2img, use only text-based")
    
    args = parser.parse_args()

    rejected_path = os.path.join(args.output, "rejected_concepts.json")

    if args.stage == "generate_only":
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index,
                            use_naive_prompt=args.naive_prompt,
                            no_image_cond=args.no_image_cond)

    elif args.stage == "verify_base":
        stage_verify_base(args.database, args.output)

    elif args.stage == "refine":
        stage_refine(args.database, rejected_path, args.output)

    elif args.stage == "final_judge":
        stage_final_judge(args.database, args.output)

    elif args.stage == "full_auto":
        print("\n🚀 STARTING FULL AUTO PIPELINE")
        stage_generate_only(args.database, args.output,
                            args.num_shards, args.shard_index)
        rejected_path = stage_verify_base(args.database, args.output)
        stage_refine(args.database, rejected_path, args.output)
        stage_final_judge(args.database, args.output)
        print("\n🏁 PIPELINE COMPLETED")