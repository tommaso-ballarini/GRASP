import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoImageProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    _QWEN3_AVAILABLE = True
except ImportError:
    _QWEN3_AVAILABLE = False


class ModelInterface:
    """Base interface for different language models."""

    MINICPM_PATH = "openbmb/MiniCPM-o-2_6"

    def __init__(self, model_path, device="cuda", attn_implementation="sdpa", torch_dtype=torch.bfloat16):
        self.device = device
        self.model_path = model_path
        self.tokenizer = None
        self.processor = None
        self.image_processor = None
        self._model_type = None

        if model_path == self.MINICPM_PATH:
            self._init_minicpm(model_path, attn_implementation, torch_dtype)
        elif "qwen" in model_path.lower() and _QWEN3_AVAILABLE:
            self._init_qwen3vl(model_path, attn_implementation, torch_dtype)
        else:
            raise ValueError(f"Unsupported model path: '{model_path}'.")

    def _init_minicpm(self, model_path, attn_implementation, torch_dtype):
        self._model_type = "minicpm"
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            attn_implementation=attn_implementation, torch_dtype=torch_dtype
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _init_qwen3vl(self, model_path, attn_implementation, torch_dtype):
        self._model_type = "qwen3vl"
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

    def chat(self, msgs):
        if self._model_type == "qwen3vl":
            return self._chat_qwen3vl(msgs)
        elif self._model_type == "minicpm":
            return self._chat_minicpm(msgs)
        else:
            raise NotImplementedError(f"chat() not implemented for: {self._model_type}")

    def _chat_qwen3vl(self, msgs: list) -> tuple:
        text_input = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt", padding=True,
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = inputs.input_ids.shape[1]
        # Token IDs grezzi dei soli token generati (non il prompt)
        generated_ids = output.sequences[:, prompt_len:]

        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # ConfidenceCalculator si aspetta:
        # outputs["sequences"][0] → lista di token IDs (tensor 1D)
        # outputs["scores"][i][0] → logit step i (tensor 1D, vocab_size)
        outputs_for_conf = {
            "sequences": generated_ids,   # shape: (1, n_generated_tokens)
            "scores": output.scores,      # tupla di n_generated_tokens tensori (1, vocab_size)
        }

        return outputs_for_conf, decoded
        
    def _chat_minicpm(self, msgs):
        with torch.no_grad():
            result = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        text = result[-1] if isinstance(result, tuple) else str(result)
        return {"sequences": text, "logits": None}