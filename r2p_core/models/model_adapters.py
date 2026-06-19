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

class MiniCPMAdapter(ModelAdapter):
    """Adapter for MiniCPM model message format."""

    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        return [{'role': 'user', 'content': [test_image, ret_image, prompt]}]

    def format_text_options_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

    def format_personalized_caption_msgs(self, test_image, prompt):
        return [{'role': 'user', 'content': [test_image, prompt]}]

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
class LLaVANeXTInterleaveAdapter(ModelAdapter):
    """Adapter for LLaVA-NeXT Interleave model message format."""
    
    def __init__(self, model, tokenizer, image_processor, model_config):
        """
        Initialize the LLaVA-NeXT Interleave adapter.
        
        Args:
            model: The LLaVA model
            tokenizer: The tokenizer for the model
            image_processor: The image processor for the model
            model_config: The configuration of the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.use_im_start_end = (hasattr(model_config, "mm_use_im_start_end") and 
                                model_config.mm_use_im_start_end)
    
    def _format_images_in_prompt(self, images):
        """
        Format multiple images in a prompt according to LLaVA-NeXT Interleave format.
        
        Args:
            images: List of images
            
        Returns:
            str: Formatted image tokens string
        """
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        
        formatted_images = ""
        
        if self.use_im_start_end:
            # For newer LLaVA models with separate start/end tokens
            for i, _ in enumerate(images):
                # Add image token with index
                image_token = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{i}{DEFAULT_IM_END_TOKEN}"
                formatted_images += image_token + "\n"
        else:
            # For older LLaVA models or models that don't use start/end tokens
            for _ in images:
                formatted_images += f"{DEFAULT_IMAGE_TOKEN}\n"
                
        return formatted_images
    
    def format_image2image_plus_text_comparison_msgs(self, test_image, ret_image, prompt):
        """
        Format a message with two images and text for comparison.
        
        Args:
            test_image: First image
            ret_image: Second image
            prompt: Text prompt
            
        Returns:
            dict: Formatted message structure for the model
        """
        from llava.mm_utils import process_images
        import torch
        from PIL import Image
        
        # Load images if they're paths
        if isinstance(test_image, str):
            test_image = Image.open(test_image)
        if isinstance(ret_image, str):
            ret_image = Image.open(ret_image)
            
        images = [test_image, ret_image]
        image_sizes = [img.size for img in images]
        
        # Process images for the model
        image_tensors = process_images(images, self.image_processor, self.model_config)
        
        # Format the prompt with image tokens
        formatted_image_tokens = self._format_images_in_prompt(images)
        formatted_prompt = formatted_image_tokens + prompt
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": formatted_prompt}]
            conv = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            conv = formatted_prompt
            
        return {
            "prompt": conv,
            "images": image_tensors,
            "image_sizes": image_sizes
        }

    def format_text_options_msgs(self, test_image, prompt):
        """
        Format a message with a single image and text.
        
        Args:
            test_image: Image
            prompt: Text prompt
            
        Returns:
            dict: Formatted message structure for the model
        """
        from llava.mm_utils import process_images
        import torch
        from PIL import Image
        
        # Load image if it's a path
        if isinstance(test_image, str):
            test_image = Image.open(test_image)
            
        images = [test_image]
        image_sizes = [test_image.size]
        
        # Process image for the model
        image_tensors = process_images(images, self.image_processor, self.model_config)
        
        # Format the prompt with image token
        formatted_image_tokens = self._format_images_in_prompt(images)
        formatted_prompt = formatted_image_tokens + prompt
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": formatted_prompt}]
            conv = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            conv = formatted_prompt
            
        return {
            "prompt": conv,
            "images": image_tensors,
            "image_sizes": image_sizes
        }

    def format_personalized_caption_msgs(self, test_image, prompt):
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": test_image}, {"type": "text", "text": prompt}],
            }
        ]
    def format_attribute_based_text_options_msgs(self, test_image, prompt):
        """
        Format a message with a single image and attribute-focused text.
        This is similar to format_text_options_msgs but kept separate for semantic clarity.
        
        Args:
            test_image: Image
            prompt: Text prompt
            
        Returns:
            dict: Formatted message structure for the model
        """
        # For LLaVA-NeXT, the implementation is the same as format_text_options_msgs
        return self.format_text_options_msgs(test_image, prompt)
        
    def generate_response(self, formatted_input, max_new_tokens=1024, temperature=0.7):
        """
        Generate a response using the LLaVA-NeXT model based on the formatted input.
        
        Args:
            formatted_input: Input dict created by one of the format_* methods
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: The model's response
        """
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        import torch
        
        # Extract inputs
        prompt = formatted_input["prompt"]
        image_tensors = formatted_input["images"]
        image_sizes = formatted_input.get("image_sizes")
        
        # Tokenize the prompt
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        
        # Set up image tensors correctly for the model
        if isinstance(image_tensors, list):
            num_patches_im0 = image_tensors[0].size(0)
            num_patches_im1 = image_tensors[1].size(0)
            if num_patches_im0 > num_patches_im1:
                im_tensor_0 = image_tensors[0][:num_patches_im1, :, :]
                im_tensor_1 = image_tensors[1]
            else:
                im_tensor_0 = image_tensors[0]
                im_tensor_1 = image_tensors[1][:num_patches_im0, :, :]
            image_tensors = torch.cat([im_tensor_0, im_tensor_1], dim=0)
        image_tensors = image_tensors.to(self.model.device, dtype=torch.float16)
        if hasattr(self.tokenizer, 'apply_chat_template'):
            # For models that support chat template
            messages = [{"role": "user", "content": prompt}]
            conv = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template support
            conv = formatted_prompt
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                images=image_tensors,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                image_sizes=image_sizes,
                return_dict_in_generate=True,
                output_logits=True
            )
        output_ids, logits = outputs.sequences, outputs.logits
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return outputs, response

# Example usage:
"""
# Initialize model components
tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name)
model_config = model.config

# Create adapter
adapter = LLaVANeXTInterleaveAdapter(model, tokenizer, image_processor, model_config)

# Format inputs for a comparison task
formatted_input = adapter.format_image2image_plus_text_comparison_msgs(
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "Compare these two images and tell me the main differences."
)

# Generate response
response = adapter.generate_response(formatted_input)
print(response)
"""