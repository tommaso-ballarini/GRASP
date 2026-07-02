# refine.py
"""
Iterative Refinement Loop con Feedback VLM (verify.py V5)

ARCHITETTURA MEMORY-EFFICIENT:
==============================
Per GPU con 24GB (L4), usiamo "model swapping":
- FASE GENERAZIONE: Carica SDXL (~8GB) → genera → scarica
- FASE VERIFICA: Carica MiniCPM+CLIP (~18GB) → verifica → scarica
- Ripeti se necessario

Questo approccio è più lento ma permette di usare modelli grandi
su GPU limitate. In letteratura: "Model Swapping" o "Sequential Loading".

MODELLI USATI:
- verify.py (verify_generation_r2p) → MiniCPM + CLIP (guida refinement)
- Final Judge → Chiamare separatamente da judge.py (Qwen2.5-VL)

LETTERATURA:
- Negative Prompt refinement: CFG (Ho & Salimans, 2022), SDXL (Podell, 2023)
- VQA-based verification: TIFA (Hu et al., ICCV 2023)
- Iterative refinement: Self-Correcting Diffusion (Feng, 2024)
- Model Swapping: Common in production with limited VRAM
"""

import os
import sys
import json
import gc
import torch
from pathlib import Path
from PIL import Image

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from pipeline.utils2 import cleanup_gpu, ensure_output_dir, get_iteration_filename
from pipeline.verify import verify_generation_r2p


def _full_cleanup():
    """Force complete VRAM cleanup between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    cleanup_gpu()


def _load_generator(temp_db_path: str, output_dir: str):
    """Load SDXL + IP-Adapter generator."""
    from pipeline.generate import Generator
    
    generator = Generator(
        database_path=temp_db_path,
        output_dir=output_dir
    )
    
    if not generator._initialize_pipeline():
        return None
    
    return generator


def _unload_generator(generator):
    """Completely unload generator from VRAM."""
    if generator is None:
        return
    
    if hasattr(generator, 'pipe') and generator.pipe is not None:
        # Unload all pipeline components
        if hasattr(generator.pipe, 'to'):
            try:
                generator.pipe.to('cpu')
            except:
                pass
        del generator.pipe
        generator.pipe = None
    
    del generator
    _full_cleanup()


def _load_verifier():
    """Load MiniCPM + CLIP for verification."""
    from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
    from pipeline.r2p_tools import ClipScoreCalculator
    
    print("   📦 Loading MiniCPM Reasoner...")
    reasoner = MiniCPMReasoning(
        model_path=Config.VLM_MODEL,
        device="cuda",
        torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
        attn_implementation="sdpa",
        seed=Config.SEED
    )
    
    print("   📦 Loading CLIP Calculator...")
    clip_calculator = ClipScoreCalculator(device="cuda")
    
    return reasoner, clip_calculator


def _unload_verifier(reasoner, clip_calculator):
    """Completely unload verifier models from VRAM."""
    if reasoner is not None:
        if hasattr(reasoner, 'model'):
            del reasoner.model
        if hasattr(reasoner, 'tokenizer'):
            del reasoner.tokenizer
        del reasoner
    
    if clip_calculator is not None:
        if hasattr(clip_calculator, 'model'):
            del clip_calculator.model
        if hasattr(clip_calculator, 'processor'):
            del clip_calculator.processor
        del clip_calculator
    
    _full_cleanup()


def iterative_refinement(
    reference_image_path: str,
    fingerprints_dict: dict,
    output_dir: str = "output",
    # Loop parameters
    max_iterations: int = None,
    target_accuracy: float = None,
    min_improvement: float = None,
    # Verify V5 parameters (passed to verify_generation_r2p)
    vlm_high_confidence: float = 0.85,
    vlm_low_confidence: float = 0.40,
    worst_k_threshold: float = 0.50,
):
    """
    Loop di raffinamento iterativo con feedback VLM (verify.py V5).
    
    MEMORY-EFFICIENT: Usa "model swapping" per GPU con 24GB.
    Carica SDXL per generare, poi lo scarica e carica MiniCPM per verificare.
    
    Questo modulo è SOLO il loop di refinement. Il Final Judge va
    chiamato separatamente dopo usando judge.py.
    
    Args:
        reference_image_path: Immagine target da riprodurre
        fingerprints_dict: Attributi estratti (Fingerprint List)
        output_dir: Cartella output
        max_iterations: Override Config.MAX_ITERATIONS
        target_accuracy: Override Config.TARGET_ACCURACY  
        min_improvement: Override Config.MIN_IMPROVEMENT
        vlm_high_confidence: Soglia alta per auto-pass (V5)
        vlm_low_confidence: Soglia bassa per auto-fail (V5)
        worst_k_threshold: Soglia per worst-k detection (V5)
    
    Returns:
        dict: {
            "best_image": path,
            "best_score": float,
            "is_verified": bool,
            "failed_attributes": list,
            "iterations": int,
            "verification": dict (ultimo verify result),
            "history": [...]
        }
    """
    # Config with overrides
    max_iter = max_iterations or Config.MAX_ITERATIONS
    target_acc = target_accuracy or Config.TARGET_ACCURACY
    min_impr = min_improvement or Config.MIN_IMPROVEMENT
    
    print(f"\n{'='*60}")
    print(f"🔁 ITERATIVE REFINEMENT LOOP (verify V5)")
    print(f"{'='*60}")
    print(f"   Reference: {Path(reference_image_path).name}")
    print(f"   Max Iterations: {max_iter}")
    print(f"   Target Accuracy: {target_acc:.0%}")
    print(f"\n   📋 Architecture:")
    print(f"      Refinement: MiniCPM + CLIP (verify_generation_r2p)")
    print(f"      Final Judge: Call judge.py SEPARATELY after this!")
    print(f"\n   🧠 Memory Mode: MODEL SWAPPING (for 24GB GPU)")
    print(f"      → SDXL loaded only during generation")
    print(f"      → MiniCPM+CLIP loaded only during verification")
    
    ensure_output_dir(output_dir)
    
    # === CREATE TEMP DATABASE ===
    temp_db = {
        "concept_dict": {
            "refine_target": {
                "name": Path(reference_image_path).stem,
                "image": [reference_image_path],
                "info": fingerprints_dict
            }
        }
    }
    temp_db_path = os.path.join(output_dir, "_temp_refine_db.json")
    with open(temp_db_path, 'w') as f:
        json.dump(temp_db, f)
    
    # === STATE VARIABLES ===
    best_image_path = None
    best_score = 0.0
    best_verification = None
    iteration = 0
    
    # Negative prompt dinamico (Prompt Reweighting - CFG/SDXL literature)
    negative_additions = []
    
    # Storia iterazioni
    history = []
    
    # Get SDXL prompt
    sdxl_prompt = fingerprints_dict.get("sdxl_prompt", fingerprints_dict.get("description", ""))
    if not sdxl_prompt:
        print("   ⚠️  No SDXL prompt found, using generic")
        sdxl_prompt = "high quality product photo"
    
    # === REFINEMENT LOOP ===
    while iteration < max_iter:
        iteration += 1
        
        print(f"\n{'─'*60}")
        print(f"🔄 ITERATION {iteration}/{max_iter}")
        print(f"{'─'*60}")
        
        # ═══════════════════════════════════════════════════════
        # PHASE A: GENERATION (load SDXL, generate, unload)
        # ═══════════════════════════════════════════════════════
        output_filename = get_iteration_filename(
            os.path.join(output_dir, f"candidate_{Path(reference_image_path).stem}.png"),
            iteration
        )
        
        # Costruisci negative prompt cumulativo (letteratura: Prompt Reweighting)
        current_negative = Config.NEGATIVE_PROMPT
        if negative_additions:
            current_negative += ", " + ", ".join(negative_additions[-5:])  # Keep last 5
            print(f"   🚫 Negatives: {', '.join(negative_additions[-3:])}")
        
        generator = None
        try:
            print(f"   📦 Loading SDXL + IP-Adapter...")
            generator = _load_generator(temp_db_path, output_dir)
            
            if generator is None:
                print("   ❌ Failed to initialize SDXL pipeline")
                break
            
            # Load reference image
            ref_img = Image.open(reference_image_path).convert("RGB")
            ref_img = ref_img.resize(
                (Config.REFERENCE_IMAGE_SIZE, Config.REFERENCE_IMAGE_SIZE), 
                Image.Resampling.LANCZOS
            )
            
            # Generate with SDXL + IP-Adapter
            print(f"   🎨 Generating candidate...")
            result = generator.pipe(
                prompt=sdxl_prompt,
                negative_prompt=current_negative,
                ip_adapter_image=ref_img,
                num_inference_steps=Config.NUM_INFERENCE_STEPS,
                guidance_scale=Config.GUIDANCE_SCALE,
                height=Config.OUTPUT_IMAGE_SIZE,
                width=Config.OUTPUT_IMAGE_SIZE,
                generator=torch.Generator(device=Config.DEVICE).manual_seed(Config.SEED + iteration)
            ).images[0]
            
            result.save(output_filename)
            print(f"   ✅ Generated: {Path(output_filename).name}")
            
            # CRITICAL: Unload SDXL before loading verifier
            print(f"   📤 Unloading SDXL to free VRAM...")
            _unload_generator(generator)
            generator = None
            
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            # Try to cleanup anyway
            if generator is not None:
                try:
                    _unload_generator(generator)
                except:
                    pass
            _full_cleanup()
            break
        
        # ═══════════════════════════════════════════════════════
        # PHASE B: VERIFICATION (load MiniCPM+CLIP, verify, unload)
        # ═══════════════════════════════════════════════════════
        print(f"   ⚖️  Verifying with MiniCPM + CLIP (V5)...")
        
        reasoner = None
        clip_calculator = None
        try:
            reasoner, clip_calculator = _load_verifier()
            
            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=output_filename,
                ref_image_path=reference_image_path,
                fingerprints=fingerprints_dict,
                vlm_high_confidence=vlm_high_confidence,
                vlm_low_confidence=vlm_low_confidence,
                worst_k_vlm_threshold=worst_k_threshold
            )
            
            # CRITICAL: Unload verifier before next iteration
            print(f"   📤 Unloading verifier to free VRAM...")
            _unload_verifier(reasoner, clip_calculator)
            reasoner = None
            clip_calculator = None
            
        except Exception as e:
            print(f"   ❌ Verification error: {e}")
            import traceback
            traceback.print_exc()
            if reasoner is not None or clip_calculator is not None:
                try:
                    _unload_verifier(reasoner, clip_calculator)
                except:
                    pass
            _full_cleanup()
            # Continue with partial result
            verification = {"score": 0, "is_verified": False, "failed_attributes": [], "method": "error"}
        
        # Extract results
        current_score = verification["score"]
        is_verified = verification["is_verified"]
        failed_attrs = verification.get("failed_attributes", [])
        method = verification.get("method", "unknown")
        
        print(f"\n   📊 Score: {current_score:.2f} | Verified: {is_verified} | Method: {method}")
        
        # Save to history
        history.append({
            "iteration": iteration,
            "image": output_filename,
            "score": current_score,
            "is_verified": is_verified,
            "method": method,
            "failed_attributes": failed_attrs.copy() if failed_attrs else [],
            "reason": verification.get("reason", "")
        })
        
        # === UPDATE BEST ===
        improvement = current_score - best_score
        
        if current_score > best_score:
            best_score = current_score
            best_image_path = output_filename
            best_verification = verification
            print(f"   🏆 NEW BEST! Score: {best_score:.2f} (+{improvement:.2f})")
        else:
            print(f"   📉 No improvement ({improvement:+.2f})")
        
        # === EXIT CONDITIONS ===
        
        # Success: verification passed (is_verified from V5)
        if is_verified:
            print(f"\n   ✅ VERIFIED! Method: {method}")
            break
        
        # Convergence: no significant improvement
        if iteration > 1 and improvement < min_impr:
            print(f"\n   🔻 Convergence: improvement {improvement:.3f} < {min_impr}")
            break
        
        # No failed attributes to fix (edge case)
        if not failed_attrs:
            print(f"\n   ✅ No failed attributes detected")
            break
        
        # === PROMPT REWEIGHTING (letteratura: CFG, SDXL) ===
        if iteration < max_iter:
            print(f"\n   🔧 Prompt Reweighting for iteration {iteration + 1}:")
            print(f"      Failed attributes: {len(failed_attrs)}")
            
            # Build negative prompts from failed attributes
            new_negatives = build_negative_from_failed(failed_attrs)
            
            for neg in new_negatives:
                if neg not in negative_additions:
                    negative_additions.append(neg)
                    print(f"      + Negative: '{neg}'")
    
    # === CLEANUP ===
    _full_cleanup()
    
    # Remove temp database
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)
    
    # === FINAL REPORT ===
    final_failed = best_verification.get("failed_attributes", []) if best_verification else []
    
    print(f"\n{'='*60}")
    print(f"🏁 REFINEMENT COMPLETED")
    print(f"{'='*60}")
    print(f"   Best Image: {Path(best_image_path).name if best_image_path else 'None'}")
    print(f"   Best Score: {best_score:.2f}")
    print(f"   Verified: {best_verification.get('is_verified', False) if best_verification else False}")
    print(f"   Iterations: {iteration}")
    print(f"   Method: {best_verification.get('method', 'N/A') if best_verification else 'N/A'}")
    
    if final_failed:
        print(f"\n   ❌ Failed Attributes ({len(final_failed)}):")
        for attr in final_failed[:5]:
            print(f"      • {attr[:50]}...")
        if len(final_failed) > 5:
            print(f"      ... and {len(final_failed) - 5} more")
    
    print(f"\n   💡 Next step: Call judge.py for independent Final Judge!")
    print(f"{'='*60}")
    
    return {
        "best_image": best_image_path,
        "best_score": best_score,
        "is_verified": best_verification.get("is_verified", False) if best_verification else False,
        "failed_attributes": final_failed,
        "iterations": iteration,
        "verification": best_verification,
        "history": history
    }


def build_negative_from_failed(failed_attributes: list) -> list:
    """
    Costruisce negative prompts da attributi falliti (dal verify V5).
    
    Letteratura:
    - Negative Prompt Guidance (Ho & Salimans, 2022)
    - SDXL Negative prompting (Podell, 2023)
    
    Args:
        failed_attributes: Lista di stringhe attributo mancanti
        
    Returns:
        Lista di stringhe per negative prompt
    """
    negatives = []
    
    for attr in failed_attributes:
        attr_lower = attr.lower()
        
        # Pattern matching per tipi comuni di attributi
        if "color" in attr_lower or any(c in attr_lower for c in ["red", "blue", "green", "black", "white"]):
            negatives.append(f"wrong color, incorrect hue")
        elif "brand" in attr_lower or "logo" in attr_lower or "text" in attr_lower:
            negatives.append("without brand logo, missing text, incorrect branding")
        elif "material" in attr_lower or any(m in attr_lower for m in ["leather", "plastic", "metal", "fabric"]):
            negatives.append(f"wrong material texture")
        elif "pattern" in attr_lower or any(p in attr_lower for p in ["stripe", "check", "solid"]):
            negatives.append("wrong pattern")
        elif "shape" in attr_lower:
            negatives.append("wrong shape, incorrect form")
        else:
            # Generic: quote the missing attribute
            negatives.append(f"missing {attr[:30]}")
    
    # Deduplicate
    return list(dict.fromkeys(negatives))


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n📋 Refine Module - Iterative Refinement with verify V5")
    print("─" * 60)
    print("""
USAGE:
    from pipeline.refine import iterative_refinement
    from pipeline.judge import FinalJudge
    
    # Step 1: Iterative Refinement
    result = iterative_refinement(
        reference_image_path="data/target.jpg",
        fingerprints_dict={
            "color": "blue",
            "material": "leather",
            "sdxl_prompt": "(blue leather bag:1.3), ..."
        },
        output_dir="output/refinement"
    )
    
    # Step 2: Final Judge (SEPARATE - modular)
    if result["best_image"]:
        judge = FinalJudge()  # Uses Qwen2.5-VL (different from MiniCPM!)
        judge_result = judge.evaluate(
            generated_image=result["best_image"],
            reference_image="data/target.jpg",
            fingerprints=fingerprints_dict,
            prompt=fingerprints_dict.get("sdxl_prompt", "")
        )
        
        print(f"Refinement passed: {result['is_verified']}")
        print(f"Final Judge passed: {judge_result.passed}")

FLOW:
    1. Iterative refinement with MiniCPM + CLIP (verify V5)
    2. Dynamic negative prompt updates from failed_attributes
    3. Exit when is_verified=True or convergence
    4. Call FinalJudge separately for independent evaluation

MEMORY MODE:
    This module uses MODEL SWAPPING for 24GB GPUs:
    - SDXL (~8GB) loaded/unloaded for each generation
    - MiniCPM+CLIP (~18GB) loaded/unloaded for each verification
    Slower but memory-safe!
    """)
