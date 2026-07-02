# pipeline/judge.py
"""
Final Judge for the R2P-GEN pipeline.

IMPORTANT: The Final Judge uses a DIFFERENT model than verify.py!
This to ensures an unbiased evaluation: the model that guided refinement is
not the same one that judges the final result.

Final evaluation with:
1. VQA-based attribute verification (TIFA-style) with InternVL3_5-8B
2. Quantitative metrics (CLIP-I, CLIP-T, DINO-I)
3. Final aggregated score with interpretable breakdown

Usage:
    from pipeline.judge import FinalJudge

    judge = FinalJudge()  # Uses InternVL3_5-8B (different from MiniCPM/Qwen3!)
    result = judge.evaluate(
        generated_image="output/generated.png",
        reference_image="data/target.jpg",
        fingerprints={"color": "blue", "material": "leather", ...},
        prompt="(blue leather bag:1.3), ..."
    )
    print(f"Final Score: {result.final_score:.2%}")
"""

import os
import sys
import json 
import torch
from PIL import Image
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from pipeline.metrics import MetricsCalculator, MetricsResult
from pipeline.utils2 import cleanup_gpu


# ============================================================================
# DEFAULT MODELS - Final Judge uses DIFFERENT model than verify/refine!
# ============================================================================
# verify.py + refine.py loop → MiniCPM / Qwen3-VL (Config.VLM_MODEL)
# judge.py (Final Judge)    → InternVL3_5-8B (independent evaluation)


@dataclass
class JudgeResult:
    """Evaluation result from the Final Judge — three raw metrics, no aggregation."""
    clip_i: float = 0.0
    clip_t: float = 0.0
    dino_i: float = 0.0
    tifa_score: float = 0.0
    attributes_present: List[str] = field(default_factory=list)
    attributes_missing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "metrics": {
                "clip_i":       self.clip_i,
                "clip_t":       self.clip_t,
                "dino_i":       self.dino_i,
                "tifa_score":   self.tifa_score,
            },
            "attributes_present": self.attributes_present,
            "attributes_missing": self.attributes_missing,
        }


class FinalJudge:
    """
    Defines the Final Judge for R2P-GEN, using InternVL3_5-8B for VQA/TIFA evaluation.
    
    """

    def __init__(
        self,
        device: str = None,
        use_dino: bool = True,
        use_clip: bool = True,
        use_vqa: bool = True,
        vlm_model_path: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "eager",
    ):
        """
        Initialize the Final Judge with InternVL3_5-8B.

        Args:
            device: Compute device ('cuda' or 'cpu')
            threshold: Minimum final_score to pass (default: Config.TARGET_ACCURACY)
            use_dino: Whether to compute DINO-I metric
            use_clip: Whether to compute CLIP-I/T metrics
            use_vqa: Whether to use VLM for VQA evaluation
            vlm_model_path: Path to InternVL3_5-8B (default: DEFAULT_JUDGE_MODEL_PATH)
            torch_dtype: dtype per il modello (default bfloat16)
            attn_implementation: 'flash_attention_2' o 'sdpa'
        """
        self.device = device or Config.DEVICE
        self.use_dino = use_dino
        self.use_clip = use_clip
        self.use_vqa = use_vqa
        self.vlm_model_path = vlm_model_path or Config.Models.JUDGE_MODEL
        self._torch_dtype = torch_dtype
        self._attn_implementation = attn_implementation

        self._reasoner = None
        self._reasoner_cls = None
        self._metrics_calc = None

        print(f"⚖️  [JUDGE] Initialized Final Judge (Independent Evaluator)")
        print(f"   Model: InternVL3_5-8B @ {self.vlm_model_path}")
        print(f"   Metrics: CLIP={use_clip}, DINO={use_dino}, VQA={use_vqa}")

    # ========================================================================
    # LAZY LOADING
    # ========================================================================

    @property
    def metrics_calc(self) -> MetricsCalculator:
        """Lazy load metrics calculator."""
        if self._metrics_calc is None:
            self._metrics_calc = MetricsCalculator(device=self.device)
        return self._metrics_calc

    @property
    def reasoner(self):
        """Lazy load InternVL3_5Reasoning.

        Returns the InternVL3_5Reasoning instance (or None if use_vqa=False).
        """
        if self._reasoner is None and self.use_vqa:
            print(f"   📦 Loading Final Judge VLM: InternVL3_5-8B...")
            print(f"      (Independent from MiniCPM/Qwen3 used in verify/refine)")
            try:
                from r2p_core.models.internvl_reasoning import InternVL3_5Reasoning
                self._reasoner = InternVL3_5Reasoning(
                    model_path=self.vlm_model_path,
                    device=self.device,
                    torch_dtype=self._torch_dtype,
                    attn_implementation=self._attn_implementation,
                )
                # FIX: store the class reference — the import above is local to
                # this property and is NOT visible in other methods
                # (e.g. _vqa_evaluate_attribute), which therefore cannot call
                # "InternVL3_5Reasoning._resize(...)" directly.
                self._reasoner_cls = InternVL3_5Reasoning
                print(f"      ✅ InternVL3_5-8B loaded successfully")
            except Exception as e:
                print(f"   ❌ InternVL3_5-8B load failed: {e}")
                self._reasoner = None

        return self._reasoner

    # ========================================================================
    # VQA EVALUATION (TIFA-style)
    # ========================================================================

    def _vqa_evaluate_attribute(self, image: Image.Image, question: str) -> Dict:
        reasoner = self.reasoner
        if reasoner is None:
            return {"is_present": False, "raw_response": "VLM not available"}

        image = self._reasoner_cls._resize(image, max_dim=896)
        prompt_text = f"{question} Answer ONLY 'Yes' or 'No'."
        formatted = reasoner.adapter.format_attribute_based_text_options_msgs(image, prompt_text)
        output = reasoner.model_interface.chat(formatted)

        response = output.get("sequences", "").strip()
        is_present = response.lower().strip().startswith("yes")

        return {"is_present": is_present, "raw_response": response[:100]}

    def _evaluate_tifa(
        self,
        image: Union[str, Image.Image],
        fingerprints: Dict
    ) -> tuple:
        """
        TIFA-style evaluation: generate questions from fingerprints and evaluate.

        Args:
            image: Generated image to evaluate
            fingerprints: Dict of attribute -> value

        Returns:
            (tifa_score, present_list, missing_list, vqa_responses)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if max(image.size) > Config.MAX_IMAGE_DIM:
            ratio = Config.MAX_IMAGE_DIM / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        questions = self.metrics_calc.generate_tifa_questions(fingerprints)

        if not questions:
            print("   ⚠️  No valid attributes to verify")
            return 0.0, [], [], {}

        print(f"   🔍 Evaluating {len(questions)} attributes via VQA (InternVL3_5)...")

        present = []
        missing = []
        responses = {}

        for q in questions:
            attr = q["attribute"]
            question = q["question"]

            try:
                result = self._vqa_evaluate_attribute(image, question)
                responses[attr] = result

                if result["is_present"]:
                    present.append(attr)
                    print(f"      ✅ {attr}")
                else:
                    missing.append(attr)
                    print(f"      ❌ {attr}: {result['raw_response'][:50]}")

            except Exception as e:
                import traceback
                print(f"      ⚠️  Error on {attr}: {e!r}")
                traceback.print_exc()
                missing.append(attr)
                responses[attr] = {"error": str(e)}

        total = len(questions)
        tifa_score = len(present) / total if total > 0 else 0.0

        return tifa_score, present, missing, responses

    # ========================================================================
    # MAIN EVALUATION
    # ========================================================================

    def evaluate(
        self,
        generated_image: Union[str, Image.Image],
        reference_image: Union[str, Image.Image],
        fingerprints: Dict,
        prompt: str = None
    ) -> JudgeResult:
        """
        Complete evaluation of a generated image.

        Args:
            generated_image: Path or PIL Image of generated output
            reference_image: Path or PIL Image of reference/target
            fingerprints: Dict of attributes to verify
            prompt: Prompt used (for CLIP-T, optional)

        Returns:
            JudgeResult with all metrics and pass/fail decision
        """
        print(f"\n⚖️  [JUDGE] Final Evaluation")
        print(f"{'─'*50}")

        result = JudgeResult()

        # === CLIP Metrics ===
        if self.use_clip:
            print("   📊 Computing CLIP metrics...")
            result.clip_i = self.metrics_calc.compute_clip_i(generated_image, reference_image)
            print(f"      CLIP-I (identity): {result.clip_i:.3f}")

            if prompt:
                result.clip_t = self.metrics_calc.compute_clip_t(generated_image, prompt)
                print(f"      CLIP-T (prompt):   {result.clip_t:.3f}")

        # === DINO Metrics ===
        if self.use_dino:
            print("   📊 Computing DINO metrics...")
            try:
                result.dino_i = self.metrics_calc.compute_dino_i(generated_image, reference_image)
                print(f"      DINO-I (details):  {result.dino_i:.3f}")
            except Exception as e:
                print(f"      ⚠️  DINO failed: {e}")
                result.dino_i = 0.0

        # === VQA/TIFA Metrics ===
        if self.use_vqa:
            print("   📊 Computing VQA/TIFA metrics (InternVL3_5-8B)...")
            tifa, present, missing, _ = self._evaluate_tifa(generated_image, fingerprints)
            result.tifa_score = tifa
            result.attributes_present = present
            result.attributes_missing = missing
            print(f"      TIFA Score: {tifa:.1%} ({len(present)}/{len(present)+len(missing)})")

        # === Summary ===
        print(f"\n{'─'*50}")
        print(f"   CLIP-I: {result.clip_i:.3f}  |  "
              f"DINO-I: {result.dino_i:.3f}  |  "
              f"TIFA: {result.tifa_score:.1%}")
        print(f"{'─'*50}")

        return result

    def quick_evaluate(
        self,
        generated_image: Union[str, Image.Image],
        reference_image: Union[str, Image.Image],
        prompt: str = None
    ) -> float:
        """
        Quick evaluation using only CLIP metrics (no VLM loading).

        Returns:
            float: Quick score based on CLIP-I (and CLIP-T if prompt provided)
        """
        clip_i = self.metrics_calc.compute_clip_i(generated_image, reference_image)

        if prompt:
            clip_t = self.metrics_calc.compute_clip_t(generated_image, prompt)
            return 0.6 * clip_i + 0.4 * clip_t

        return clip_i

    def cleanup(self):
        """Release all GPU memory."""
        if self._reasoner is not None:
            self._reasoner.cleanup()
            self._reasoner = None
            self._reasoner_cls = None

        if self._metrics_calc is not None:
            self._metrics_calc.cleanup()
            self._metrics_calc = None

        cleanup_gpu()
        print("   🧹 Judge resources released")


# ============================================================================
# STANDALONE USAGE
# ============================================================================

if __name__ == "__main__":
    judge = FinalJudge(
        threshold=0.85,
        use_dino=True,
        use_clip=True,
        use_vqa=True,
    )

    test_fingerprints = {
        "category": "bag",
        "color": "blue",
        "material": "leather",
        "pattern": "solid",
        "brand/text": "none"
    }

    print("\n📋 Example usage:")
    print("   judge = FinalJudge()")
    print("   result = judge.evaluate(gen_img, ref_img, fingerprints, prompt)")
    print("   if result.passed: ...")