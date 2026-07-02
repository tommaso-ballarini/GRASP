import torch


class ModelInterface:
    """Base interface for different language models."""

    def __init__(self, model_path, device="cuda", attn_implementation="sdpa", torch_dtype=torch.bfloat16):
        self.device = device
        self.model_path = model_path
        self.tokenizer = None
        self.processor = None
        self.image_processor = None

    def chat(self, msgs):
        raise NotImplementedError("Subclasses must implement chat().")