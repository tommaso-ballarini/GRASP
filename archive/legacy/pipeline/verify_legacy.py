import torch
import os
import re
import numpy as np
from PIL import Image
from pipeline.r2p_tools import ClipScoreCalculator, LLMConfidenceExtractor, MiniCPMAdapter

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _extract_attributes_for_clip(fingerprints: dict) -> list:
    """Estrae gli attributi puliti mantenendo l'ordine di priorità."""
    attributes = []
    keys_to_check = ['distinct features', 'brand/text', 'pattern', 'material', 'color']
    seen = set()
    
    for key in keys_to_check:
        if key in fingerprints and fingerprints[key]:
            val = fingerprints[key]
            if "no visible" in val.lower() or "none" in val.lower(): continue
            
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

# =============================================================================
# MAIN LOGIC V5: WORST-K + CLIP-FIRST + FAILED ATTRIBUTES
# =============================================================================

def verify_generation_r2p(
    reasoner, 
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
    R2P Verification Pipeline V5.
 
    
    Pipeline:
        Phase 0: CLIP Quick Reject
        Phase 1: VLM Single Check (attribute sweep on generated image)
        Phase 2: CLIP Detailed Analysis
        Phase 3: Decision Gate (with Worst-K override)
        Phase 4: Pairwise Reasoning (with Worst-K=1 check)
        
    Returns:
        dict with keys:
            - is_verified: bool
            - score: float (final confidence score)
            - method: str (decision method used)
            - reason: str (human-readable explanation)
            - failed_attributes: list[str] (attributes that failed, for refine phase)
            - vlm_history: list[dict] (detailed VLM query history)
            - clip_details: dict (CLIP scores per attribute)
    """
    
    vlm_history = []
    clip_details = {'gen': {}, 'ref': {}}
    failed_attributes = []  # NEW: track failed attrs for refine
    
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
            "failed_attributes": []
        }

    # Extract attributes
    attributes_list = _extract_attributes_for_clip(fingerprints)
    
    if not attributes_list:
        print("⚠️ No attributes found in fingerprints!")
        return {
            "is_verified": False,
            "score": 0.0,
            "reason": "No attributes to verify",
            "vlm_history": [],
            "method": "Error_NoAttributes",
            "failed_attributes": []
        }
    
    print(f"📋 Verifying {len(attributes_list)} attributes")
    
    # =========================================================================
    # PHASE 0: CLIP QUICK REJECT (Skip VLM if obviously bad)
    # =========================================================================
    print(f"\n[Phase 0] CLIP Quick Reject Check...")
    # Compute CLIP scores for generated image and each attribute (crossmodal attribute verification of original R2P paper)
    
    gen_clip_score, gen_breakdown = clip_calculator.compute_attribute_score(gen_image, attributes_list)
    _, ref_breakdown = clip_calculator.compute_attribute_score(ref_image, attributes_list)
    clip_details = {'gen': gen_breakdown, 'ref': ref_breakdown}
    
    if gen_clip_score < clip_quickreject_floor:
        print(f"   ❌ QUICK REJECT: CLIP score {gen_clip_score:.3f} < {clip_quickreject_floor}")
        print(f"   Skipping expensive VLM queries.")
        
        # All attributes are considered failed
        failed_attributes = attributes_list.copy()
        
        return {
            "is_verified": False,
            "score": gen_clip_score,
            "method": "QuickReject_CLIP",
            "reason": f"CLIP score too low ({gen_clip_score:.3f}). VLM skipped.",
            "failed_attributes": failed_attributes,
            "vlm_history": [],
            "clip_details": clip_details
        }
    
    print(f"   ✓ CLIP score {gen_clip_score:.3f} >= {clip_quickreject_floor}, proceeding with VLM")

    # =========================================================================
    # PHASE 1: VLM SINGLE CHECK (Attribute Sweep with Real Logits)
    # =========================================================================
    print(f"\n[Phase 1] VLM Sweep on {len(attributes_list)} attributes...")
    
    # Initialize the LLM-level confidence extractor (THE FIX!)
    conf_extractor = LLMConfidenceExtractor(reasoner.model_interface)

    vlm_scores = []
    worst_k_triggered = False  # Track if any single attr is very bad

    for attr in attributes_list:
        prompt = (
            f"Look at the image. Is the feature '{attr}' clearly visible and correct?\n"
            f"Ignore minor style issues, focus on the presence of the feature.\n"
            f"Answer strictly Yes or No."
        )
        
        # Query with confidence extraction (using LLM-level interception)
        confidence, response_text = conf_extractor.query_with_confidence(
            image=gen_image,
            prompt=prompt,
            max_new_tokens=5
        )
        
        score = confidence['yes_confidence']
        vlm_scores.append(score)
        
        vlm_history.append({
            "phase": "single_check",
            "attribute": attr,
            "prompt": prompt,
            "response": response_text,
            "score": score,
            "method": confidence.get('method', 'unknown'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence']
        })
        
        # Check Worst-K=1: if any single attribute is below threshold
        if score < worst_k_vlm_threshold:
            worst_k_triggered = True
            failed_attributes.append(attr)
        
        # Status emoji based on confidence
        if score > 0.7:
            emoji = "🟢"
        elif score > worst_k_vlm_threshold:
            emoji = "🟡"
        else:
            emoji = "🔴"  # Will trigger Worst-K
            
        method_indicator = f"[{confidence.get('method', '?')}]"
        print(f"   {attr[:30]:<30} | Y:{score:.2f} N:{confidence['no_confidence']:.2f} {emoji} {method_indicator}")

    # Calculate VLM average
    vlm_avg_score = sum(vlm_scores) / len(vlm_scores) if vlm_scores else 0.0
    vlm_min_score = min(vlm_scores) if vlm_scores else 0.0
    
    # Log summary
    logit_count = sum(1 for h in vlm_history if h.get('method') == 'logits')
    print(f"\n📊 VLM Summary:")
    print(f"   Average: {vlm_avg_score:.4f} | Min: {vlm_min_score:.4f}")
    print(f"   Logit method: {logit_count}/{len(vlm_history)}")
    if worst_k_triggered:
        print(f"   ⚠️ WORST-K TRIGGERED: {len(failed_attributes)} attrs below {worst_k_vlm_threshold}")

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
        if ref_val > 0.18:  # Only check significant attributes
            gen_val = gen_breakdown.get(attr, 0)
            delta = gen_val - ref_val
            if delta < max_drop_threshold:
                clip_pass = False
                clip_issues.append(f"Drop in '{attr[:20]}' ({delta:+.3f})")
                # Add to failed if not already there
                if attr not in failed_attributes:
                    failed_attributes.append(attr)

    print(f"   CLIP Avg: {gen_clip_score:.4f} | {'PASS ✓' if clip_pass else 'FAIL ✗'}")
    if clip_issues:
        for issue in clip_issues[:3]:
            print(f"   - {issue}")
        if len(clip_issues) > 3:
            print(f"   ... and {len(clip_issues) - 3} more issues")

    # =========================================================================
    # PHASE 3: DECISION GATE (with Worst-K override)
    # =========================================================================
    print(f"\n[Phase 3] Decision Gate...")
    
    # Case 1: Strong agreement - AUTO-PASS (but NOT if Worst-K triggered!)
    if vlm_avg_score >= vlm_high_confidence and clip_pass and not worst_k_triggered:
        print("   ✅ AUTO-PASS: Strong Agreement (VLM High + CLIP Pass + No Worst-K)")
        return {
            "is_verified": True,
            "score": vlm_avg_score,
            "method": "AutoPass_Agreement",
            "reason": "VLM and CLIP strongly agree on quality.",
            "failed_attributes": [],  # No failures
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }

    # Case 2: Strong agreement - AUTO-FAIL
    if vlm_avg_score <= vlm_low_confidence and not clip_pass:
        print("   ❌ AUTO-FAIL: Strong Agreement (VLM Low + CLIP Fail)")
        return {
            "is_verified": False,
            "score": vlm_avg_score,
            "method": "AutoFail_Agreement",
            "reason": f"Both models rejected. CLIP issues: {clip_issues[:3]}",
            "failed_attributes": failed_attributes,
            "vlm_history": vlm_history,
            "clip_details": clip_details
        }
    
    # Case 3: Disagreement OR Worst-K triggered - go to Pairwise
    reason_for_trigger = []
    if worst_k_triggered:
        reason_for_trigger.append(f"Worst-K: {len(failed_attributes)} attrs below {worst_k_vlm_threshold}")
    if not clip_pass: 
        reason_for_trigger.append("CLIP detected drops")
    if vlm_avg_score < vlm_high_confidence: 
        reason_for_trigger.append(f"VLM uncertain ({vlm_avg_score:.2f})")
    
    print(f"   ⚠️ REQUIRES ARBITRATION: {', '.join(reason_for_trigger)}")
    print("   >>> TRIGGERING PHASE 4 (Pairwise Analysis)")

    # =========================================================================
    # PHASE 4: PAIRWISE REASONING (with Worst-K=1 final check)
    # =========================================================================
    print(f"\n[Phase 4] Pairwise Comparison on {len(attributes_list)} attributes...")
    
    pairwise_scores = []
    pairwise_failed = []  # Track pairwise failures
    
    for attr in attributes_list:
        prompt_text = (
            f"Compare the specific feature '{attr}' in Image 1 (generated) and Image 2 (reference).\n"
            f"Is this feature present and visually consistent in both images?\n"
            f"Answer Yes or No."
        )
        
        # Pairwise query with both images
        confidence, response_text = conf_extractor.query_with_confidence(
            image=gen_image,
            prompt=prompt_text,
            image2=ref_image,
            max_new_tokens=5
        )
        
        p_yes = confidence['yes_confidence']
        pairwise_scores.append(p_yes)
        
        vlm_history.append({
            "phase": "pairwise",
            "attribute": attr,
            "score": p_yes,
            "method": confidence.get('method', 'unknown'),
            "yes_conf": confidence['yes_confidence'],
            "no_conf": confidence['no_confidence']
        })
        
        # Check Worst-K=1 for pairwise
        if p_yes < worst_k_pairwise_threshold:
            pairwise_failed.append(attr)
            if attr not in failed_attributes:
                failed_attributes.append(attr)
        
        emoji = "✅" if p_yes > 0.6 else "⚠️" if p_yes >= worst_k_pairwise_threshold else "❌"
        method_indicator = f"[{confidence.get('method', '?')}]"
        print(f"   {attr[:25]:<25} | Y:{p_yes:.2f} {emoji} {method_indicator}")

    # Final decision
    final_pairwise_score = np.mean(pairwise_scores) if pairwise_scores else 0.0
    pairwise_min_score = min(pairwise_scores) if pairwise_scores else 0.0
    
    print(f"\n📊 Pairwise Summary:")
    print(f"   Average: {final_pairwise_score:.4f} | Min: {pairwise_min_score:.4f}")
    
    # Decision logic: Check Worst-K=1 first, then mean threshold
    if pairwise_failed:
        # Worst-K=1 failure: any attribute below threshold = FAIL
        is_verified = False
        method = "Fail_WorstK_Pairwise"
        reason = f"Pairwise Worst-K fail: {len(pairwise_failed)} attrs below {worst_k_pairwise_threshold}"
        print(f"   ❌ FAIL (Worst-K): {pairwise_failed[:3]}...")
    elif final_pairwise_score >= pairwise_mean_threshold:
        is_verified = True
        method = "Pass_Pairwise"
        reason = f"Pairwise verification passed. Score: {final_pairwise_score:.2f}"
        failed_attributes = []  # Clear failures on pass
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
        "clip_details": clip_details
    }