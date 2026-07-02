"""
R2P-GEN Full Pipeline Orchestrator

This module orchestrates the complete R2P-GEN pipeline:
1. Build Database (fingerprint extraction)
2. Generate (image generation with SDXL + IP-Adapter)
3. Verify (generated image verification with V5 logit-based confidence)
4. Refine (iterative refinement loop)

The pipeline can run in two modes:
- Full Database Mode: Process all images from a database JSON
- Single Image Mode: Process a single target image with optional refinement
"""

import os
import json
import torch
import gc
from pathlib import Path

from pipeline.build_database import DatabaseBuilder
from pipeline.generate import Generator
from pipeline.verify import verify_generation_r2p
from pipeline.refine import iterative_refinement
from pipeline.judge import FinalJudge
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.utils2 import cleanup_gpu, ensure_output_dir
from r2p_core.models.mini_cpm_reasoning import MiniCPMReasoning
from config import Config


def run_database_pipeline(
    database_path: str,
    output_dir: str = None,
    skip_generation: bool = False,
    skip_verification: bool = False
):
    """
    Run the full R2P-GEN pipeline on a pre-built database.
    
    This is the main batch processing mode:
    1. Load database (built by build_database.py)
    2. Generate images for all concepts
    3. Verify each generated image
    
    Args:
        database_path: Path to fingerprint database JSON
        output_dir: Output directory (default: from Config)
        skip_generation: If True, skip generation (use existing images)
        skip_verification: If True, skip verification step
        
    Returns:
        dict: Results with generation and verification stats
    """
    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN DATABASE PIPELINE")
    print(f"{'='*70}")
    print(f"   Database: {database_path}")
    print(f"   Output: {output_dir or Config.OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(database_path):
        print(f"❌ Database not found: {database_path}")
        print("   Run: python pipeline/build_database.py first")
        return None
    
    output_dir = output_dir or Config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    
    results = {
        "database_path": database_path,
        "output_dir": output_dir,
        "generation_stats": None,
        "verification_results": {}
    }
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: GENERATION
    # ═══════════════════════════════════════════════════════
    if not skip_generation:
        print(f"{'─'*70}")
        print("📍 PHASE 1: IMAGE GENERATION")
        print(f"{'─'*70}")
        
        generator = Generator(
            database_path=database_path,
            output_dir=output_dir
        )
        
        generation_stats = generator.generate_all()
        results["generation_stats"] = generation_stats
        
        generator.cleanup()
        cleanup_gpu()
    else:
        print("⏭️  Skipping generation (using existing images)")
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: VERIFICATION
    # ═══════════════════════════════════════════════════════
    if not skip_verification:
        print(f"\n{'─'*70}")
        print("📍 PHASE 2: IMAGE VERIFICATION (V5)")
        print(f"{'─'*70}")
        
        # Load database for verification
        with open(database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
        
        concept_dict = database.get("concept_dict", {})
        
        # Initialize verification models
        print("   Loading verification models...")
        reasoner = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
        clip_calculator = ClipScoreCalculator(device="cuda")
        
        verified_count = 0
        failed_count = 0
        
        for concept_id, content in concept_dict.items():
            # Find generated image
            gen_image_path = os.path.join(output_dir, f"{concept_id}_generated.png")
            
            if not os.path.exists(gen_image_path):
                print(f"   ⚠️  {concept_id}: Generated image not found, skipping")
                continue
            
            # Get reference image (first selected image)
            selected_images = content.get("selected_images", [])
            if not selected_images:
                print(f"   ⚠️  {concept_id}: No reference images, skipping")
                continue
            
            ref_image_path = selected_images[0]
            fingerprints = content.get("info", {})
            
            # Run verification
            verification = verify_generation_r2p(
                reasoner=reasoner,
                clip_calculator=clip_calculator,
                gen_image_path=gen_image_path,
                ref_image_path=ref_image_path,
                fingerprints=fingerprints
            )
            
            results["verification_results"][concept_id] = verification
            
            if verification["is_verified"]:
                verified_count += 1
                print(f"   ✅ {concept_id}: PASS ({verification['method']})")
            else:
                failed_count += 1
                print(f"   ❌ {concept_id}: FAIL ({verification['method']})")
        
        # Cleanup
        del reasoner
        del clip_calculator
        cleanup_gpu()
        
        # Summary
        total = verified_count + failed_count
        print(f"\n📊 Verification Summary: {verified_count}/{total} passed ({100*verified_count/max(1,total):.1f}%)")
    else:
        print("⏭️  Skipping verification")
    
    return results


def run_single_image_pipeline(
    target_image_path: str,
    fingerprints: dict = None,
    output_dir: str = None,
    use_refinement: bool = True,
    use_final_judge: bool = True
):
    """
    Run the R2P-GEN pipeline on a single target image.
    
    This mode is useful for:
    - Testing with a single image
    - Iterative refinement experiments
    - Quick prototyping
    
    Pipeline:
    1. (Optional) Extract fingerprints if not provided
    2. Generate + Verify + Refine (MiniCPM)
    3. (Optional) Final Judge (Qwen2.5-VL - DIFFERENT model!)
    
    Args:
        target_image_path: Path to the target image
        fingerprints: Pre-extracted fingerprints (optional, will extract if None)
        output_dir: Output directory
        use_refinement: If True, use iterative refinement loop
        use_final_judge: If True, run Final Judge after refinement (Qwen2.5-VL)
        
    Returns:
        dict: Result with best_image, best_score, verification details, judge result
    """
    print(f"\n{'='*70}")
    print(f"🚀 R2P-GEN SINGLE IMAGE PIPELINE")
    print(f"{'='*70}")
    print(f"   Target: {target_image_path}")
    print(f"   Mode: {'Iterative Refinement' if use_refinement else 'Single-Shot'}")
    print(f"   Final Judge: {'Enabled (Qwen2.5-VL)' if use_final_judge else 'Disabled'}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(target_image_path):
        print(f"❌ Error: File {target_image_path} not found.")
        return None
    
    output_dir = output_dir or Config.OUTPUT_DIR
    ensure_output_dir(output_dir)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 1: EXTRACTION (if fingerprints not provided)
    # ═══════════════════════════════════════════════════════
    if fingerprints is None:
        print(f"{'─'*70}")
        print("📍 PHASE 1: FINGERPRINT EXTRACTION (MiniCPM)")
        print(f"{'─'*70}")
        
        # Use DatabaseBuilder's extract_single_image method
        builder = DatabaseBuilder(
            source_dir=".",  # Not used for single image
            output_path=os.path.join(output_dir, "_temp.json"),
            device=Config.DEVICE,
            model_path=Config.VLM_MODEL
        )
        
        fingerprints = builder.extract_single_image(target_image_path)
        
        # CRITICAL: Full cleanup to free VRAM before generation
        # MiniCPM extraction uses ~16GB, need to free it completely
        if builder.extractor is not None:
            del builder.extractor.model
            del builder.extractor.tokenizer
            del builder.extractor.clip_model
            del builder.extractor
            builder.extractor = None
        del builder
        
        # Force garbage collection and CUDA cache clear
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        cleanup_gpu()
        
        print("   🧹 Cleaned up extraction model (freed VRAM)")
        
        if fingerprints is None:
            print("❌ Failed to extract fingerprints")
            return None
        
        print(f"   ✅ Extracted {len(fingerprints)} attributes")
        print(f"   📝 SDXL Prompt: {fingerprints.get('sdxl_prompt', '')[:80]}...")
    else:
        print(f"✅ Using provided fingerprints ({len(fingerprints)} keys)")
    
    # ═══════════════════════════════════════════════════════
    # PHASE 2: GENERATION & VERIFICATION (MiniCPM)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'─'*70}")
    print("📍 PHASE 2: GENERATION & VERIFICATION (MiniCPM + CLIP)")
    print(f"{'─'*70}")
    
    if use_refinement:
        # Iterative refinement with MiniCPM + CLIP (verify V5)
        result = iterative_refinement(
            reference_image_path=target_image_path,
            fingerprints_dict=fingerprints,
            output_dir=output_dir
        )
        
        final_image = result.get("best_image")
        final_score = result.get("best_score", 0)
        is_verified = result.get("is_verified", False)
        iterations_used = result.get("iterations", 0)
        
    else:
        # Single-shot mode
        from pipeline.generate import Generator
        
        # Create temporary database for single image
        # Generator expects: name, info (with sdxl_prompt), image (list)
        temp_db = {
            "concept_dict": {
                "single_target": {
                    "name": "single_target",
                    "image": [target_image_path],
                    "info": fingerprints
                }
            }
        }
        
        temp_db_path = os.path.join(output_dir, "_temp_single_db.json")
        with open(temp_db_path, 'w') as f:
            json.dump(temp_db, f)
        
        # Generate
        generator = Generator(
            database_path=temp_db_path,
            output_dir=output_dir
        )
        generator.generate_all()
        generator.cleanup()
        
        # Filename pattern: {name}_ipa_{method_suffix}.png
        method_suffix = "layerwise" if Config.USE_LAYERWISE_SCALING else "global"
        final_image = os.path.join(output_dir, f"single_target_ipa_{method_suffix}.png")
        
        # Verify
        print("   Loading verification models...")
        reasoner = MiniCPMReasoning(
            model_path=Config.VLM_MODEL,
            device="cuda",
            torch_dtype=torch.float16 if Config.USE_FP16 else torch.float32,
            attn_implementation="sdpa",
            seed=Config.SEED
        )
        clip_calculator = ClipScoreCalculator(device="cuda")
        
        verification = verify_generation_r2p(
            reasoner=reasoner,
            clip_calculator=clip_calculator,
            gen_image_path=final_image,
            ref_image_path=target_image_path,
            fingerprints=fingerprints
        )
        
        final_score = verification["score"]
        is_verified = verification["is_verified"]
        iterations_used = 1
        
        # Cleanup
        del reasoner
        del clip_calculator
        cleanup_gpu()
        os.remove(temp_db_path)  # Remove temp database
        
        result = {
            "best_image": final_image,
            "best_score": final_score,
            "is_verified": is_verified,
            "iterations": 1,
            "verification": verification
        }
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3: FINAL JUDGE (Qwen2.5-VL - DIFFERENT MODEL!)
    # ═══════════════════════════════════════════════════════
    judge_result = None
    final_passed = None
    
    if use_final_judge and final_image:
        print(f"\n{'─'*70}")
        print("📍 PHASE 3: FINAL JUDGE (Qwen2.5-VL - Independent)")
        print(f"{'─'*70}")
        print("   Model: Qwen2.5-VL (DIFFERENT from MiniCPM used in refinement!)")
        print("   Metrics: CLIP-I, CLIP-T, DINO-I, TIFA")
        
        judge = FinalJudge(
            threshold=Config.TARGET_ACCURACY,
            use_dino=True,
            use_clip=True,
            use_vqa=True
        )
        
        try:
            judge_eval = judge.evaluate(
                generated_image=final_image,
                reference_image=target_image_path,
                fingerprints=fingerprints,
                prompt=fingerprints.get("sdxl_prompt", "")
            )
            
            judge_result = judge_eval.to_dict()
            final_passed = judge_eval.passed
            
            print(f"\n   📊 Final Judge Results:")
            print(f"      CLIP-I: {judge_eval.clip_i:.3f}")
            print(f"      CLIP-T: {judge_eval.clip_t:.3f}")
            print(f"      DINO-I: {judge_eval.dino_i:.3f}")
            print(f"      TIFA:   {judge_eval.tifa_score:.1%}")
            print(f"      Final Score: {judge_eval.final_score:.2%}")
            print(f"      Passed: {'✅ YES' if final_passed else '❌ NO'}")
            
        except Exception as e:
            print(f"   ⚠️  Final Judge error: {e}")
            import traceback
            traceback.print_exc()
            final_passed = None
        
        finally:
            judge.cleanup()
        
        # Add to result
        result["judge_result"] = judge_result
        result["final_passed"] = final_passed
    
    # ═══════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("🏁 PIPELINE COMPLETED")
    print(f"{'='*70}")
    print(f"   📁 Input:  {Path(target_image_path).name}")
    print(f"   📁 Output: {Path(final_image).name if final_image else 'None'}")
    print(f"   � Iterations: {iterations_used}")
    
    print(f"\n   📊 Results by Stage:")
    print(f"      Refinement (MiniCPM): {'✅ PASS' if is_verified else '❌ FAIL'} (Score: {final_score:.2f})")
    
    if use_final_judge:
        if final_passed is None:
            print(f"      Final Judge (Qwen): ⚠️  ERROR or NOT RUN")
        elif final_passed:
            print(f"      Final Judge (Qwen): ✅ PASS")
        else:
            print(f"      Final Judge (Qwen): ❌ FAIL")
    
    # Overall success requires BOTH stages to pass
    overall_success = is_verified and (final_passed if use_final_judge else True)
    
    if overall_success:
        print(f"\n   ✅ SUCCESS - All stages passed!")
    elif is_verified and not use_final_judge:
        print(f"\n   ✅ SUCCESS - Refinement passed (Final Judge disabled)")
    elif is_verified:
        print(f"\n   ⚠️  PARTIAL - Refinement passed, Final Judge failed")
    else:
        print(f"\n   ❌ FAILURE - Refinement did not verify")
    
    print(f"{'='*70}\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="R2P-GEN Full Pipeline")
    parser.add_argument("--mode", choices=["database", "single"], default="database",
                       help="Pipeline mode: 'database' for batch, 'single' for one image")
    parser.add_argument("--database", type=str, default="database/database_perva_train_1_clip.json",
                       help="Path to fingerprint database (for database mode)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to target image (for single mode)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip generation (use existing images)")
    parser.add_argument("--skip-verification", action="store_true",
                       help="Skip verification step")
    parser.add_argument("--no-refinement", action="store_true",
                       help="Disable iterative refinement (single mode only)")
    
    args = parser.parse_args()
    
    if args.mode == "database":
        run_database_pipeline(
            database_path=args.database,
            output_dir=args.output,
            skip_generation=args.skip_generation,
            skip_verification=args.skip_verification
        )
    else:
        if not args.image:
            print("❌ --image is required for single mode")
            print("   Usage: python full_loop.py --mode single --image path/to/image.jpg")
        else:
            # Single image mode - fingerprints extracted automatically!
            result = run_single_image_pipeline(
                target_image_path=args.image,
                fingerprints=None,  # Will be extracted automatically via DatabaseBuilder
                output_dir=args.output or "output/single",
                use_refinement=not args.no_refinement,
                use_final_judge=True  # Enable final judge by default
            )
            
            if result:
                print(f"\n📁 Output saved to: {result.get('best_image', 'N/A')}")