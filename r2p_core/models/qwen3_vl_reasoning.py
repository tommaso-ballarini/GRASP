import torch
import numpy as np
import random
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from .model_interface import ModelInterface
from .model_adapters import QwenAdapter
from r2p_core.evaluators.compute_confidence import ConfidenceCalculator
from qwen_vl_utils import process_vision_info  # compatible with Qwen3-VL (same vision message schema as Qwen2.5-VL)
from .prompt_generator import BasePromptGenerator


class Qwen3VLModel(ModelInterface):
    """
    Qwen3-VL model wrapper.
    Mirrors QwenModel's interface (chat returns {"sequences", "logits"} + decoded text)
    so that ConfidenceCalculator works unchanged.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        torch_dtype=torch.bfloat16,
        enable_thinking: bool = False,
    ):
        super().__init__(model_path, device, attn_implementation, torch_dtype)
        self.enable_thinking = enable_thinking
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def _apply_template(self, msgs: list) -> str:
        """
        Wrap apply_chat_template with a fallback for processor versions
        that do not yet expose enable_thinking (avoids TypeError on older builds).
        enable_thinking=False is important: if Qwen3 thinks out loud with <think>…</think>
        tokens, ConfidenceCalculator would look for the "answer" trigger in the middle
        of the reasoning trace and produce wrong confidence scores.
        """
        try:
            return self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Older processor build — thinking mode not exposed via template kwarg.
            # Thinking is off by default in Qwen3-VL inference, so this is safe.
            return self.processor.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

    def chat(self, msgs: list):
        """
        Run a forward + greedy/sampled decode pass.

        Returns
        -------
        outputs : dict  {"sequences": list[Tensor], "logits": tuple[Tensor]}
            Shape matches what ConfidenceCalculator.calculate_confidence expects.
        output_text : str
            Decoded answer string (special tokens stripped).
        """
        
        text = self._apply_template(msgs)
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        raw_outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,   # scores invece di output_logits
            do_sample=False,      # greedy per confidence stabile
        )

        # Token IDs solo dei token generati (escluso il prompt)
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = raw_outputs.sequences[:, prompt_len:]  # shape: (1, n_tokens)

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # Formato atteso da ConfidenceCalculator:
        # sequences: tensore (1, n_tokens) di IDs
        # scores: tupla di n_tokens tensori (1, vocab_size) — uno per step generato
        outputs = {
            "sequences": generated_ids,      # tensore, non lista
            "scores": raw_outputs.scores,    # tupla, non logits
        }

        return outputs, output_text

class Qwen3VLReasoning:
    """
    Drop-in replacement for MiniCPMReasoning (and QwenReasoning) in:
      - pipeline/verify.py  (stage_verify_base)
      - pipeline/recovery.py (post-cure medical check)

    Public API is intentionally identical to QwenReasoning so callers
    only need to swap the import + instantiation.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        torch_dtype=torch.bfloat16,
        seed: int = 42,
        enable_thinking: bool = False,
    ):
        print(f"Loading Qwen3-VL Reasoning with model {model_path}")
        self._set_seed(seed)
        self.device = device
        self.model_interface = Qwen3VLModel(
            model_path, device, attn_implementation, torch_dtype, enable_thinking
        )
        # QwenAdapter message format is forward-compatible with Qwen3-VL
        # ({"type": "image", "image": ...} / {"type": "text", "text": ...} schema is unchanged)
        self.adapter = QwenAdapter(self.model_interface.processor)
        self.conf_calculator = ConfidenceCalculator(self.model_interface.processor.tokenizer)
        self.prompt_generator = BasePromptGenerator()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _chat(self, msgs, task, target_tokens=None):
        outputs, answer = self.model_interface.chat(msgs)
        confidence = self.conf_calculator.calculate_confidence(
            outputs, task, target_tokens=target_tokens
        )
        return confidence, answer

    @staticmethod
    def _resize(image, max_long_side: int):
        """
        Resize a PIL image so its longest side equals max_long_side,
        preserving aspect ratio.
        PIL.Image.resize takes (width, height) — note the original QwenReasoning
        had the axes swapped (it passed (max_height, new_width) which PIL
        interprets as width=max_height, height=new_width). Fixed here.
        """
        w, h = image.size
        scale = max_long_side / max(w, h)
        return image.resize((int(w * scale), int(h * scale)))

    # ------------------------------------------------------------------
    # Public reasoning methods (same signatures as QwenReasoning)
    # ------------------------------------------------------------------

    def generate_personalized_caption(self, test_image, concept_name, category_name, answer_format):
        answer_format = {"Caption": "<caption>"}
        prompt = self.prompt_generator.get_personalized_caption_prompt(
            test_image, concept_name, category_name, answer_format
        )
        msgs = self.adapter.format_personalized_caption_msgs(test_image, prompt)
        _, caption = self.model_interface.chat(msgs)
        if "sorry" in caption:
            caption = f"A photo of {concept_name}"
        return caption

    def reason_with_multiple_text(
        self, test_image, test_question, descriptions, task, answer_format, target_tokens
    ):
        if "A" in answer_format:
            prompt = self.prompt_generator.get_attribute_based_text_options_prompt(
                test_question, descriptions, answer_format
            )
        else:
            prompt = self.prompt_generator.get_text_options_prompt(
                test_question, descriptions, answer_format
            )
        img = self._resize(test_image, 800)
        msgs = self.adapter.format_text_options_msgs(img, prompt)
        confidence, answer = self._chat(msgs, task, target_tokens=target_tokens)
        return confidence, answer, None

    def reason_with_only_text(
        self, test_image, test_question, descriptions, task, answer_format, target_tokens=None
    ):
        prompt = self.prompt_generator.get_text_options_prompt(
            test_question, descriptions, answer_format
        )
        img = self._resize(test_image, 800)
        msgs = self.adapter.format_text_options_msgs(img, prompt)
        confidence, answer = self._chat(msgs, task, target_tokens=target_tokens)
        return confidence, answer, None

    def reason_image2image_plus_text(
        self, test_image, ret_image, test_question, descriptions, task, answer_format
    ):
        prompt = self.prompt_generator.get_image2image_plus_text_comparison_prompt(
            test_question, descriptions, answer_format
        )
        test_img = self._resize(test_image, 1000)
        ret_img = self._resize(ret_image, 1000)
        msgs = self.adapter.format_image2image_plus_text_comparison_msgs(
            test_img, ret_img, prompt
        )
        confidence, answer = self._chat(msgs, task)
        return confidence, answer, None