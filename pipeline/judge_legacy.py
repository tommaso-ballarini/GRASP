# pipeline/judge.py
"""
Final Judge per R2P-GEN Pipeline.

IMPORTANTE: Il Final Judge usa un modello DIVERSO da verify.py!
- verify.py / refine.py loop → Qwen3-VL (verifica durante generazione)
- judge.py → InternVL3_5-8B (valutazione finale indipendente)

Questo garantisce una valutazione imparziale: il modello che ha guidato
il refinement NON è lo stesso che giudica il risultato finale.

Valutazione finale con:
1. VQA-based attribute verification (TIFA-style) con InternVL3_5-8B
2. Metriche quantitative (CLIP-I, CLIP-T, DINO-I)
3. Score aggregato finale con breakdown interpretabile

Usage:
    from pipeline.judge import FinalJudge

    judge = FinalJudge()  # Usa InternVL3_5-8B (diverso da MiniCPM/Qwen3!)
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
import torch
from PIL import Image
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

# Add project paths
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
    """Complete evaluation result from the Final Judge."""
    # Core scores
    final_score: float = 0.0
    passed: bool = False

    # Individual metrics
    clip_i: float = 0.0
    clip_t: float = 0.0
    dino_i: float = 0.0
    tifa_score: float = 0.0

    # VQA Details
    attributes_present: List[str] = field(default_factory=list)
    attributes_missing: List[str] = field(default_factory=list)
    vqa_responses: Dict[str, Dict] = field(default_factory=dict)

    # Metadata
    threshold_used: float = 0.0
    metrics_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "final_score": self.final_score,
            "passed": self.passed,
            "metrics": {
                "clip_i": self.clip_i,
                "clip_t": self.clip_t,
                "dino_i": self.dino_i,
                "tifa_score": self.tifa_score
            },
            "attributes_present": self.attributes_present,
            "attributes_missing": self.attributes_missing,
            "vqa_responses": self.vqa_responses,
            "threshold": self.threshold_used,
            "breakdown": self.metrics_breakdown
        }


class FinalJudge:
    """
    Final Judge per valutazione immagini generate.

    IMPORTANTE: Usa un modello DIVERSO da verify.py/refine.py!
    - verify/refine → MiniCPM / Qwen3-VL (guida il refinement)
    - FinalJudge   → InternVL3_5-8B (valutazione indipendente)

    Combina:
    - VQA con InternVL3_5-8B per verifica attributi (TIFA-style)
    - CLIP-I/T per identity e prompt faithfulness
    - DINO-I per fine-grained identity

    Attributes:
        vlm_model_path: Path a InternVL3_5-8B
        metrics_calc: Calculator for CLIP/DINO metrics
        threshold: Minimum score to pass evaluation

    Example:
        judge = FinalJudge(threshold=0.85)
        result = judge.evaluate(gen_img, ref_img, fingerprints, prompt)

        if result.passed:
            print("✅ Final approval by independent judge!")
        else:
            print(f"❌ Failed: {result.attributes_missing}")
    """

    def __init__(
        self,
        device: str = None,
        threshold: float = None,
        use_dino: bool = True,
        use_clip: bool = True,
        use_vqa: bool = True,
        vlm_model_path: str = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
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
        self.threshold = threshold or Config.TARGET_ACCURACY
        self.use_dino = use_dino
        self.use_clip = use_clip
        self.use_vqa = use_vqa
        self.vlm_model_path = vlm_model_path or Config.Models.JUDGE_MODEL
        self._torch_dtype = torch_dtype
        self._attn_implementation = attn_implementation

        # Lazy loaded
        self._reasoner = None
        self._metrics_calc = None

        print(f"⚖️  [JUDGE] Initialized Final Judge (Independent Evaluator)")
        print(f"   Model: InternVL3_5-8B @ {self.vlm_model_path}")
        print(f"   Attn:  {attn_implementation}  |  dtype: {torch_dtype}")
        print(f"   Threshold: {self.threshold:.0%}")
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
        """
        Lazy load InternVL3_5Reasoning.

        Restituisce l'istanza di InternVL3_5Reasoning (o None se use_vqa=False).
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
                print(f"      ✅ InternVL3_5-8B loaded successfully")
            except Exception as e:
                print(f"   ❌ InternVL3_5-8B load failed: {e}")
                self._reasoner = None

        return self._reasoner

    # ========================================================================
    # VQA EVALUATION (TIFA-style)
    # ========================================================================

    def _vqa_evaluate_attribute(
        self,
        image: Image.Image,
        question: str
    ) -> Dict:
        """
        Ask a Yes/No question about an image using InternVL3_5.

        Args:
            image: PIL Image to evaluate
            question: Binary question (expects Yes/No)

        Returns:
            Dict with 'answer', 'is_present', 'confidence', 'raw_response'
        """
        reasoner = self.reasoner
        if reasoner is None:
            return {
                "answer": "no",
                "is_present": False,
                "confidence": 0.0,
                "raw_response": "VLM not available"
            }

        # Ridimensiona se necessario (speculare a _resize in InternVL3_5Reasoning)
        image = InternVL3_5Reasoning._resize(image, max_dim=896)

        # Costruisce il messaggio nel formato atteso da InternVLAdapter
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": f"{question} Answer ONLY 'Yes' or 'No'."}
                ]
            }
        ]

        # Passa attraverso adapter → model_interface.chat()
        formatted = reasoner.adapter.format_messages(msgs)
        output = reasoner.model_interface.chat(formatted)

        response = output.get("sequences", "").strip()
        logits = output.get("logits", None)

        # Confidence dal ConfidenceCalculator se logits disponibili,
        # altrimenti fallback lessicale
        if logits is not None:
            confidence = reasoner.conf_calculator.from_logits(logits)
        else:
            confidence = reasoner.conf_calculator.from_text(response)

        response_lower = response.lower()
        is_present = any(
            word in response_lower[:30]
            for word in ["yes", "correct", "present", "sì", "si"]
        )

        return {
            "answer": "yes" if is_present else "no",
            "is_present": is_present,
            "confidence": float(confidence),
            "raw_response": response[:100]
        }

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

        # Resize se necessario
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
                    print(f"      ✅ {attr}  (conf: {result['confidence']:.2f})")
                else:
                    missing.append(attr)
                    print(f"      ❌ {attr}: {result['raw_response'][:50]}...")

            except Exception as e:
                print(f"      ⚠️  Error on {attr}: {e}")
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
            prompt: SDXL prompt used (for CLIP-T, optional)

        Returns:
            JudgeResult with all metrics and pass/fail decision
        """
        print(f"\n⚖️  [JUDGE] Final Evaluation")
        print(f"{'─'*50}")

        result = JudgeResult(threshold_used=self.threshold)

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
            tifa, present, missing, responses = self._evaluate_tifa(generated_image, fingerprints)
            result.tifa_score = tifa
            result.attributes_present = present
            result.attributes_missing = missing
            result.vqa_responses = responses
            print(f"      TIFA Score: {tifa:.1%} ({len(present)}/{len(present)+len(missing)})")

        # === Aggregate Final Score ===
        metrics_result = MetricsResult(
            clip_i=result.clip_i,
            clip_t=result.clip_t,
            dino_i=result.dino_i,
            tifa_score=result.tifa_score
        )
        result.final_score = self.metrics_calc.compute_final_score(metrics_result)

        result.metrics_breakdown = {
            "clip_i": result.clip_i,
            "clip_t": result.clip_t,
            "dino_i": result.dino_i,
            "tifa": result.tifa_score,
            "weighted_final": result.final_score
        }

        result.passed = result.final_score >= self.threshold

        # === Summary ===
        print(f"\n{'─'*50}")
        print(f"   🏆 FINAL SCORE: {result.final_score:.1%}")
        print(f"   {'✅ PASSED' if result.passed else '❌ FAILED'} (threshold: {self.threshold:.0%})")
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

        Useful for fast iteration during refinement loop.

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