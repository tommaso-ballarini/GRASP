"""R2P Generation Verification (Qwen3-VL).

Implements a multi-phase verification pipeline for generated images:
 0. CLIP quick-reject on extracted attributes
 1. VLM single-attribute checks (Qwen3-VL)
 2. CLIP detailed analysis vs. reference
 3. Decision gate with Worst-K logic
 4. Pairwise VQA comparison (if arbitration required)

Returns a dict containing: is_verified, score, method, reason,
failed_attributes, vlm_history, and clip_details.
"""
import os
import re
import numpy as np
from PIL import Image

from pipeline.r2p_tools import ClipScoreCalculator

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Chiavi storiche, usate per i fingerprint OBJECT (e per qualsiasi fingerprint
# senza "_entity_type" esplicito, cioè i vecchi dataset di ablation).
# NON TOCCARE: deve restare identica per non rompere run passate.
_KEYS_OBJECT = ['distinct features', 'brand/text', 'pattern', 'material', 'color']

# Chiavi per i fingerprint LIVING (schema introdotto con DreamBench).
# Usate SOLO quando fingerprints["_entity_type"] == "LIVING" esplicitamente.
_KEYS_LIVING = ['distinctive_markings', 'facial_features', 'coat_and_color', 'accessories']


def _extract_attributes_for_clip(fingerprints: dict) -> list:
    """Extract clean attributes in priority order for CLIP scoring.

    Retrocompatibilità: se "_entity_type" non è presente nel fingerprint
    (vecchi dataset di ablation, solo OBJECT) il comportamento è identico
    a prima di questa modifica — si usa SEMPRE _KEYS_OBJECT.
    Le chiavi LIVING vengono usate solo se "_entity_type" == "LIVING" è
    esplicitamente dichiarato nel fingerprint (schema DreamBench).
    """
    attributes = []

    entity_type = fingerprints.get("_entity_type")  # None se assente (vecchi dataset)
    if entity_type == "LIVING":
        keys_to_check = _KEYS_LIVING
    else:
        keys_to_check = _KEYS_OBJECT

    seen = set()

    for key in keys_to_check:
        if key in fingerprints and fingerprints[key]:
            val = fingerprints[key]
            
            # --- INIZIO FIX ---
            # Se 'val' è una lista (es. ['red', 'blue']), uniscila in una singola stringa separata da virgole
            if isinstance(val, list):
                val = ", ".join([str(v) for v in val])
            # Se 'val' non è una stringa (es. un dizionario o un intero), forzalo a stringa
            elif not isinstance(val, str):
                val = str(val)
            # --- FINE FIX ---

            if "no visible" in val.lower() or "none" in val.lower():
                continue
            chunks = re.split(r'[;.]|\n', val)
            for chunk in chunks:
                clean_chunk = chunk.strip()
                if len(clean_chunk) > 4 and "image" not in clean_chunk.lower():
                    if clean_chunk not in seen:
                        attributes.append(clean_chunk)
                        seen.add(clean_chunk)

    if not attributes and 'attributes' in fingerprints:
        raw_attrs = []
        if isinstance(fingerprints['attributes'], list):
            raw_attrs = [str(v) for v in fingerprints['attributes']]
        elif isinstance(fingerprints['attributes'], dict):
            raw_attrs = [str(v) for v in fingerprints['attributes'].values()]
        for attr in raw_attrs:
            if attr not in seen:
                attributes.append(attr)
                seen.add(attr)

    return attributes

def _resize(image: Image.Image, max_long_side: int) -> Image.Image:
    """
    Resize a PIL image so the longest side equals max_long_side,
    preserving aspect ratio.
    PIL.Image.resize takes (width, height) in that order.
    """
    w, h = image.size
    scale = max_long_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)))


def _query_single(reasoner, image: Image.Image, prompt: str):
    """
    Single-image yes/no query through Qwen3VLReasoning internals.

    The prompt must instruct the model to respond with JSON {"answer": "yes/no"}
    so ConfidenceCalculator.extract_candidate_indices can locate the trigger token
    "answer" and read the following yes/no logit.

    Returns
    -------
    confidence : dict   {"yes_confidence", "no_confidence", "yes_score",
                         "no_score", "margin", ...}
    response_text : str  raw decoded output
    """
    msgs = reasoner.adapter.format_text_options_msgs(image, prompt)
    outputs, response_text = reasoner.model_interface.chat(msgs)
    confidence = reasoner.conf_calculator.calculate_confidence(outputs, task="recognition")
    return confidence, response_text


def _query_pairwise(reasoner, gen_image: Image.Image, ref_image: Image.Image, prompt: str):
    """
    Two-image yes/no query comparing generated image (Image 1) vs reference (Image 2).
    Same JSON-format contract as _query_single.
    """
    msgs = reasoner.adapter.format_image2image_plus_text_comparison_msgs(
        gen_image, ref_image, prompt
    )
    outputs, response_text = reasoner.model_interface.chat(msgs)
    confidence = reasoner.conf_calculator.calculate_confidence(outputs, task="recognition")
    return confidence, response_text


# =============================================================================
# MAIN LOGIC V5: WORST-K + CLIP-FIRST + FAILED ATTRIBUTES
# =============================================================================

def verify_generation_r2p(
    reasoner,                               # Qwen3VLReasoning instance
    clip_calculator: ClipScoreCalculator,
    gen_image_path: str,
    ref_image_path: str,
    fingerprints: dict,
    # CONFIDENCE THRESHOLDS
    vlm_high_confidence: float = 0.85,
    vlm_low_confidence: float = 0.40,
    clip_hard_floor: float = 0.15,
    clip_quickreject_floor: float = 0.10,
    max_drop_threshold: float = -0.03,
    # WORST-K THRESHOLDS
    worst_k_vlm_threshold: float = 0.50,
    worst_k_pairwise_threshold: float = 0.40,
    pairwise_mean_threshold: float = 0.60,
):
    """
    R2P Verification Pipeline Qwen3-VL.

    Pipeline:
        Phase 0: CLIP Quick Reject
        Phase 1: VLM Single Check (attribute sweep on generated image)
        Phase 2: CLIP Detailed Analysis
        Phase 3: Decision Gate (with Worst-K override)
        Phase 4: Pairwise Reasoning (with Worst-K=1 check)

    Parameters
    ----------
    reasoner : Qwen3VLReasoning
        Loaded reasoning wrapper. Must expose .adapter, .model_interface, .conf_calculator.
    clip_calculator : ClipScoreCalculator
    gen_image_path : str
    ref_image_path : str
    fingerprints : dict   Object fingerprint from extraction stage.

    Returns
    -------
    dict with keys:
        is_verified      : bool
        score            : float  (final confidence score)
        method           : str    (decision branch used)
        reason           : str    (human-readable explanation)
        failed_attributes: list[str]
        vlm_history      : list[dict]
        clip_details     : dict
    """
    vlm_history = []
    clip_details = {'gen': {}, 'ref': {}}
    failed_attributes = []

    print(f"\n{'='*60}")
    print(f"R2P VERIFICATION - {os.path.basename(gen_image_path)}")
    print(f"{'='*60}")

    # Load images
    try:
        gen_image = Image.open(gen_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
    except Exception as e:
        return {
            "is_verified": False,
            "score": 0.0,
            "reason": f"Image error: {e}",
            "vlm_history": [],
            "method": "Error",
            "failed_attributes": [],
            "clip_details": clip_details,
        }

    attributes_list = _extract_attributes_for_clip(fingerprints)

    if not attributes_list:
        print("⚠️  No attributes found in fingerprints!")
        return {
            "is_verified": False,
            "score": 0.0,
            "reason": "No attributes to verify",
            "vlm_history": [],
            "method": "Error_NoAttributes",
            "failed_attributes": [],
            "clip_details": clip_details,
        }

    print(f"📋 Verifying {len(attributes_list)} attributes")

    # =========================================================================
    # PHASE 0: CLIP QUICK REJECT
    # =========================================================================
    print(f"\n[Phase 0] CLIP Quick Reject Check...")
    gen_clip_score, gen_breakdown = clip_calculator.compute_attribute_score(gen_image, attributes_list)
    _, ref_breakdown = clip_calculator.compute_attribute_score(ref_image, attributes_list)
    clip_details = {'gen': gen_breakdown, 'ref': ref_breakdown}

    if gen_clip_score < clip_quickreject_floor:
        print(f"   ❌ QUICK REJECT: CLIP score {gen_clip_score:.3f} < {clip_quickreject_floor}")
        print(f"   Skipping expensive VLM queries.")
        return {
            "is_verified": False,
            "score": gen_clip_score,
            "method": "QuickReject_CLIP",
            "reason": f"CLIP score too low ({gen_clip_score:.3f}). VLM skipped.",
            "failed_attributes": attributes_list.copy(),
            "vlm_history": [],
            "clip_details": clip_details,
        }

    print(f"   ✓ CLIP score {gen_clip_score:.3f} >= {clip_quickreject_floor}, proceeding with VLM")

    # =========================================================================
    # PHASE 1: VLM SINGLE CHECK
    # Resize once before the loop — no repeated PIL ops per attribute.
    # =========================================================================
    print(f"\n[Phase 1] VLM Sweep on {len(attributes_list)} attributes...")
    gen_image_p1 = _resize(gen_image, 800)

    vlm_scores = []
    worst_k_triggered = False

    for attr in attributes_list:
        prompt = (
            f"Look at the image. Is the feature '{attr}' clearly visible and correct?\n"
            f"Ignore minor style differences; focus on whether the feature is present.\n"
            f'Respond only with JSON: {{"answer": "yes"}} or {{"answer": "no"}}.'
        )

        confidence, response_text = _query_single(reasoner, gen_image_p1, prompt)
        score = confidence['yes_confidence']
        vlm_scores.append(score)

        vlm_history.append({
            "phase": "single_check",
            "attribute": attr,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "method": confidence.get('method', 'conf_calc'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence'],
        })

        if score < worst_k_vlm_threshold:
            worst_k_triggered = True
            failed_attributes.append(attr)

        if score > 0.7:
            emoji = "🟢"
        elif score > worst_k_vlm_threshold:
            emoji = "🟡"
        else:
            emoji = "🔴"

        print(f"   {attr[:30]:<30} | Y:{score:.2f} N:{confidence['no_confidence']:.2f} {emoji}")

    vlm_avg_score = sum(vlm_scores) / len(vlm_scores) if vlm_scores else 0.0
    vlm_min_score = min(vlm_scores) if vlm_scores else 0.0

    print(f"\n📊 VLM Summary:")
    print(f"   Average: {vlm_avg_score:.4f} | Min: {vlm_min_score:.4f}")
    if worst_k_triggered:
        print(f"   ⚠️  WORST-K TRIGGERED: {len(failed_attributes)} attrs below {worst_k_vlm_threshold}")

    # =========================================================================
    # PHASE 2: CLIP DETAILED ANALYSIS
    # =========================================================================
    print(f"\n[Phase 2] CLIP Detailed Analysis...")
    clip_pass = True
    clip_issues = []

    if gen_clip_score < clip_hard_floor:
        clip_pass = False
        clip_issues.append(f"Low Avg Score ({gen_clip_score:.2f})")

    for attr, ref_val in ref_breakdown.items():
        if ref_val > 0.18:
            gen_val = gen_breakdown.get(attr, 0)
            delta = gen_val - ref_val
            if delta < max_drop_threshold:
                clip_pass = False
                clip_issues.append(f"Drop in '{attr[:20]}' ({delta:+.3f})")
                if attr not in failed_attributes:
                    failed_attributes.append(attr)

    print(f"   CLIP Avg: {gen_clip_score:.4f} | {'PASS ✓' if clip_pass else 'FAIL ✗'}")
    if clip_issues:
        for issue in clip_issues[:3]:
            print(f"   - {issue}")
        if len(clip_issues) > 3:
            print(f"   ... and {len(clip_issues) - 3} more issues")

    # =========================================================================
    # PHASE 3: DECISION GATE
    # =========================================================================
    print(f"\n[Phase 3] Decision Gate...")

    if vlm_avg_score >= vlm_high_confidence and clip_pass and not worst_k_triggered:
        print("   ✅ AUTO-PASS: Strong Agreement (VLM High + CLIP Pass + No Worst-K)")
        return {
            "is_verified": True,
            "score": vlm_avg_score,
            "method": "AutoPass_Agreement",
            "reason": "VLM and CLIP strongly agree on quality.",
            "failed_attributes": [],
            "vlm_history": vlm_history,
            "clip_details": clip_details,
        }

    if vlm_avg_score <= vlm_low_confidence and not clip_pass:
        print("   ❌ AUTO-FAIL: Strong Agreement (VLM Low + CLIP Fail)")
        return {
            "is_verified": False,
            "score": vlm_avg_score,
            "method": "AutoFail_Agreement",
            "reason": f"Both models rejected. CLIP issues: {clip_issues[:3]}",
            "failed_attributes": failed_attributes,
            "vlm_history": vlm_history,
            "clip_details": clip_details,
        }

    reason_for_trigger = []
    if worst_k_triggered:
        reason_for_trigger.append(f"Worst-K: {len(failed_attributes)} attrs below {worst_k_vlm_threshold}")
    if not clip_pass:
        reason_for_trigger.append("CLIP detected drops")
    if vlm_avg_score < vlm_high_confidence:
        reason_for_trigger.append(f"VLM uncertain ({vlm_avg_score:.2f})")

    print(f"   ⚠️  REQUIRES ARBITRATION: {', '.join(reason_for_trigger)}")
    print("   >>> TRIGGERING PHASE 4 (Pairwise Analysis)")

    # =========================================================================
    # PHASE 4: PAIRWISE REASONING
    # Resize both images once before the loop.
    # =========================================================================
    print(f"\n[Phase 4] Pairwise Comparison on {len(attributes_list)} attributes...")
    gen_image_p4 = _resize(gen_image, 1000)
    ref_image_p4 = _resize(ref_image, 1000)

    pairwise_scores = []
    pairwise_failed = []

    for attr in attributes_list:
        prompt_text = (
            f"Compare the feature '{attr}' in Image 1 (generated) and Image 2 (reference).\n"
            f"Is this feature present and visually consistent in both images?\n"
            f'Respond only with JSON: {{"answer": "yes"}} or {{"answer": "no"}}.'
        )

        confidence, response_text = _query_pairwise(
            reasoner, gen_image_p4, ref_image_p4, prompt_text
        )
        p_yes = confidence['yes_confidence']
        pairwise_scores.append(p_yes)

        vlm_history.append({
            "phase": "pairwise",
            "attribute": attr,
            "prompt": prompt_text,
            "response": response_text,
            "score": p_yes,
            "method": confidence.get('method', 'conf_calc'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence'],
        })

        if p_yes < worst_k_pairwise_threshold:
            pairwise_failed.append(attr)
            if attr not in failed_attributes:
                failed_attributes.append(attr)

        emoji = "✅" if p_yes > 0.6 else "⚠️" if p_yes >= worst_k_pairwise_threshold else "❌"
        print(f"   {attr[:25]:<25} | Y:{p_yes:.2f} {emoji}")

    final_pairwise_score = float(np.mean(pairwise_scores)) if pairwise_scores else 0.0
    pairwise_min_score = min(pairwise_scores) if pairwise_scores else 0.0

    print(f"\n📊 Pairwise Summary:")
    print(f"   Average: {final_pairwise_score:.4f} | Min: {pairwise_min_score:.4f}")

    if pairwise_failed:
        is_verified = False
        method = "Fail_WorstK_Pairwise"
        reason = f"Pairwise Worst-K fail: {len(pairwise_failed)} attrs below {worst_k_pairwise_threshold}"
        print(f"   ❌ FAIL (Worst-K): {pairwise_failed[:3]}...")
    elif final_pairwise_score >= pairwise_mean_threshold:
        is_verified = True
        method = "Pass_Pairwise"
        reason = f"Pairwise verification passed. Score: {final_pairwise_score:.2f}"
        failed_attributes = []
        print(f"   ✅ PASS: Mean {final_pairwise_score:.2f} >= {pairwise_mean_threshold}")
    else:
        is_verified = False
        method = "Fail_Pairwise_Mean"
        reason = f"Pairwise mean too low: {final_pairwise_score:.2f} < {pairwise_mean_threshold}"
        print(f"   ❌ FAIL: Mean {final_pairwise_score:.2f} < {pairwise_mean_threshold}")

    return {
        "is_verified": is_verified,
        "score": final_pairwise_score,
        "method": method,
        "reason": reason,
        "failed_attributes": failed_attributes,
        "vlm_history": vlm_history,
        "clip_details": clip_details,
    }