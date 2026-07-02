import json

class ModelAdapter:
    """Base class for model-specific message format adapters."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_text_options_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

    def format_personalized_caption_msgs(self, test_image, prompt):
        raise NotImplementedError("Subclasses must implement this method.")

class QwenAdapter(ModelAdapter):
    """Adapter for Qwen model message format."""

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image,},
                    {"type": "image", "image": ret_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def format_text_options_msgs(self, test_image, prompt):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    
    def format_personalized_caption_msgs(self, test_image, prompt):
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": test_image}, {"type": "text", "text": prompt}],
            }
        ]

class InternVLAdapter(ModelAdapter):
    """Adapter for InternVL model message format."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        """Format messages for image-to-image plus text comparison."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "image", "image": ret_image},
                    {"type": "text", "text": f"<image>\n{prompt}"}
                ]
            }
        ]

    def format_text_options_msgs(self, test_image, prompt):
        """Format messages for text options with a single image."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": f"<image>\n{prompt}"}
                ]
            }
        ]

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        """Format messages for attribute-based text options."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": test_image},
                    {"type": "text", "text": f"<image>\n{prompt}"}
                ]
            }
        ]
    
    def format_personalized_caption_msgs(self, test_image, prompt):
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": test_image}, {"type": "text", "text": prompt}],
            }
        ]
