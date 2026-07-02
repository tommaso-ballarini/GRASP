import re
import os
import torch
import glob
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import requests
from io import BytesIO
import json
import sys
sys.path.append("src/utils")
from defined import categories
from filelock import FileLock
import argparse
from transformers import CLIPModel, CLIPProcessor
sys.path.append("src/evaluators")
from compute_confidence import ClipScoreCalculator
from qwen_vl_utils import process_vision_info


def clean_and_load_json(model_output):
    if not model_output.strip():
        raise ValueError("Received empty model output!")

    # Remove markdown code block markers
    model_output = model_output.strip("```json").strip("```").strip()

    # Extract only JSON part if there's extra text
    clean_json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
    if clean_json_match:
        cleaned_json = clean_json_match.group(0)
    else:
        raise ValueError(f"Invalid JSON format: {model_output}")

    return cleaned_json


class Qwen2VLDescription:
    def __init__(self, 
                model_path="Qwen/Qwen2-VL-7B-Instruct",
                device="cuda",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16):
        """
        Initializes the Qwen2-VL model for generating object descriptions.

        Args:
            model_path (str): Hugging Face model path.
            device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
            attn_implementation (str): Attention mechanism ('flash_attention_2' or 'sdpa').
            torch_dtype (dtype): Torch data type (default: float16).
        """
        print("Loading Qwen2-VL Description Generator")
        self.device = device
        self.model_path = model_path
        
        # Load Qwen2-VL model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map="auto"
        ).eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # Load CLIP for category detection
        # self.clip_model = CLIPModel.from_pretrained(
            # 'openai/clip-vit-large-patch14-336',
            # use_safetensors=True
        # ).to(self.device)
        # self.feature_extractor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        # self.clip_scorer = ClipScoreCalculator(self.feature_extractor, self.clip_model)

    def load_image(self, image_file):
        """
        Loads an image from a file path or URL.

        Args:
            image_file (str): File path or URL of the image.

        Returns:
            PIL.Image: The loaded image.
        """
        if image_file.startswith(("http://", "https://")):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def get_detailed_input_msgs_person(self, image_test, cat, concept_identifier):
        """
        Generate prompt messages for person description.
        
        Note: This function requires 'example_database/denisdang.png' as a one-shot example.
        If the image is not available, the function will skip the one-shot example
        and use a zero-shot approach instead.
        """
        answer_format = {
            "general": "a brief description of the person in one sentence.",
            "category": "category of the person",
            "distinct features": "[List of distinct features that makes the person unique]",
        }
        
        # Check if example image exists, fallback to zero-shot if not
        example_path = 'example_database/denisdang.png'
        use_one_shot = os.path.exists(example_path)
        
        if use_one_shot:
            image_1 = self.load_image(example_path)
        else:
            print(f"⚠️ Warning: {example_path} not found. Using zero-shot approach for person.")
        
        question_example = f"""
        Describe the person in the image that is identified by the concept-identifier, <xyz> and highlight what makes him/her unique.
        Your response MUST follow EXACTLY the JSON format shown below (and nothing else):
        {json.dumps(answer_format, indent=2)}

        IMPORTANT:
        - The "general" field MUST begin with the concept-identifier. For example:
        "<xyz> is a ..."
        - DO NOT mention the clothing or accesories
        - Do not include any additional text or commentary.
        Any deviation from this format will be considered incorrect.
        """
        answer_example = (
            '{"general": "<xyz> is a young adult male with a clear, well-defined face.", '
            '"category": "Man", '
            '"distinct features": "[Silver hair, glasses, confident expression]"}'
        )
        question_test = f"""
        Describe the {cat} in the image that is identified by the concept-identifier, <{concept_identifier}> and highlight what makes him/her unique.
        Your response MUST be in valid JSON format and must follow EXACTLY the format below:
        {json.dumps(answer_format, indent=2)}

        IMPORTANT:
        - The "general" field MUST begin with "<{concept_identifier}> is ...".
        - List only the most distinguishing features that set this person apart.
        - Avoid generic descriptions that apply to every person in this category.
        - Do not include any extra text or commentary.
        Any deviation from this format will be considered incorrect.
        """

        # Build messages based on whether one-shot example is available
        if use_one_shot:
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_1},
                        {"type": "text", "text": question_example}
                    ]
                },
                {
                    "role": "assistant",
                    "content": answer_example
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_test},
                        {"type": "text", "text": question_test}
                    ]
                }
            ]
        else:
            # Zero-shot fallback
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_test},
                        {"type": "text", "text": question_test}
                    ]
                }
            ]
        return msgs

    def get_detailed_input_msgs_household(self, image_test, cat, concept_identifier):
        answer_format = {
            "general": "a brief description of the object in one sentence.",
            "category": "category of the object",
            "shape": "shape of the object",
            "material": "material",
            "color": "color of the object",
            "state": "state of the object",
            "pattern": "any distinct pattern if present",
            "distinct features": "any distinct feature that makes the object unique",
            "brand/text": "any text or brand visible on the object",
            "soft tags": "[alternative name1, name2]"
        }
        
        image_1 = self.load_image('example_database/wnr.jpg')
        question_example = f"""
Describe the plate in the image that is identified by the concept-identifier, <wnr> and highlight what makes it unique.
Your response MUST follow EXACTLY the JSON format shown below (and nothing else):
{json.dumps(answer_format, indent=2)}

IMPORTANT:
- The "general" field MUST begin with the concept-identifier. For example:
"<wnr> is a ..."
- Do not include any additional text or commentary.
Any deviation from this format will be considered incorrect.
"""
        answer_example = (
            '{"general": "<wnr> is a decorative ceramic plate with an elegant floral design around the rim", '
            '"category": "Plate", '
            '"shape": "Round with slightly raised edges", '
            '"material": "Ceramic", '
            '"state": "placed on the table", '
            '"color": "White base with orange and blue flowers and green leaves on the border.", '
            '"pattern": "Floral pattern with small, evenly spaced blossoms and foliage.", '
            '"distinct features": "The intricate detailing of the flower motifs along the edge sets it apart.", '
            '"brand/text": "No visible text or brand markings."}'
        )
        question_test = f"""
Describe the {cat} in the image that is identified by the concept-identifier, <{concept_identifier}> and highlight what makes it unique.
Your response MUST be in valid JSON format and must follow EXACTLY the format below:
{json.dumps(answer_format, indent=2)}

IMPORTANT:
- The "general" field MUST begin with "<{concept_identifier}> is ...".
- List only the most distinguishing features that set this object apart.
- Avoid generic descriptions that apply to every object in this category.
- Do not include any extra text or commentary.
Any deviation from this format will be considered incorrect.
"""

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_1},
                    {"type": "text", "text": question_example}
                ]
            },
            {
                "role": "assistant",
                "content": answer_example
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_test},
                    {"type": "text", "text": question_test}
                ]
            }
        ]
        return msgs
    
    def get_detailed_input_msgs(self, image_test, cat, concept_identifier):
        answer_format = {
            "general": "a brief description of the object in one sentence.",
            "category": "category of the object",
            "distinct features": "[List of distinct features that makes the object unique]",
        }
        
        image_1 = self.load_image('example_database/ydi.jpg')
        question_example = f"""
            Describe the bag in the image that is identified by the concept-identifier, <ydi> and highlight what makes it unique.
            Your response MUST follow EXACTLY the JSON format shown below (and nothing else):
            {json.dumps(answer_format, indent=2)}

            IMPORTANT:
            - The "general" field MUST begin with the concept-identifier. For example:
            "<ydi> is a ..."
            - Do not include any additional text or commentary.
            Any deviation from this format will be considered incorrect.
            """
        answer_example = (
            '{"general": "<ydi> is a fabric tote bag with a two-tone color scheme.", '
            '"category": "Bag", '
            '"distinct features": "[Upper half is white, lower half is yellow., Strong contrast between the two sections, Amazon style printed at the center]"}'
        )
        question_test = f"""
        Describe the {cat} in the image that is identified by the concept-identifier, <{concept_identifier}> and highlight what makes it unique.
        Your response MUST be in valid JSON format and must follow EXACTLY the format below:
        {json.dumps(answer_format, indent=2)}

        IMPORTANT:
        - The "general" field MUST begin with "<{concept_identifier}> is ...".
        - List only the most distinguishing features that set this object apart.
        - Avoid generic descriptions that apply to every object in this category.
        - Do not include any extra text or commentary.
        Any deviation from this format will be considered incorrect.
        """

        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_1},
                    {"type": "text", "text": question_example}
                ]
            },
            {
                "role": "assistant",
                "content": answer_example
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_test},
                    {"type": "text", "text": question_test}
                ]
            }
        ]
        return msgs

    def get_input_msgs_generic(self, image_test, cat, concept_identifier):
        image_example = self.load_image('example_database/bo.png')
        answer_format = {
            "info": "<a brief description of the object>",
            "category": "<category of the object in the image>"
        }
        
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_example},
                    {"type": "text", "text": f"""Describe the dog in the image that is identified by the concept-identifier, <bo>.
                    Your response must follow EXACTLY the JSON format below:
                    {json.dumps(answer_format)}
                    IMPORTANT: Your "info" field MUST begin with the <concept-identifier>. For example:
                    "<bo> is a ..."
                    Any deviation from this format will be considered incorrect."""}
                ]
            },
            {
                "role": "assistant",
                "content": '{"info": "<bo> is a well-groomed, medium-sized Shiba Inu with a thick, cinnamon-colored coat, cream accents, alert eyes, and a black collar.", "category": "dog"}'
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_test},
                    {"type": "text", "text": f"""Now, describe the {cat} in the image identified by the concept-identifier, <{concept_identifier}>.
                    Your response must follow EXACTLY the JSON format below:
                    {json.dumps(answer_format)}
                    IMPORTANT: The description in the "info" field MUST begin with "<{concept_identifier}> is ...".
                    Any deviation from this format will be considered incorrect."""}
                ]
            }
        ]
        
        return msgs

    def _generate_with_qwen(self, msgs, max_new_tokens=512):
        """
        Internal method to generate text using Qwen2-VL model.
        
        Args:
            msgs: List of message dictionaries in Qwen format
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(msgs)
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        
        # Trim input tokens and decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

    def generate_caption(self, image_file, cat, concept_identifier, args):
        """
        Generate caption for an image with optional template selection.
        
        Args:
            image_file: Path to image file
            cat: Category of object
            concept_identifier: Concept identifier tag
            args: Arguments object with template_based attribute
            
        Returns:
            str: Generated caption
        """
        # Load image
        image = self.load_image(image_file)
        
        # Auto-detect category if not provided
        cat = self._determine_category(image, cat)
        
        # Select appropriate message template
        msgs = self._select_template(image, cat, concept_identifier, args.template_based)
        
        # Generate and return caption
        output_text = self._generate_with_qwen(msgs)
        
        return output_text

    def _determine_category(self, image, cat):
        """Determine image category using CLIP if not provided."""
        if cat is not None:
            return cat
            
        # Define possible categories
        categories = [
            "person", "man", "woman", "pet animal", "plant", "building", 
            "cartoon", "clothing", "shoe", "decorative object", 
            "household object", "food item",
        ]
        
        # Use CLIP to find best matching category
        prompts = [f"A photo of a {item}" for item in categories]
        # similarities, _ = self.clip_scorer.clip_score(image, prompts)
        similarities = 0.0
        best_match_idx = torch.argmax(similarities, dim=0).item()
        
        return categories[best_match_idx]

    def _select_template(self, image, cat, concept_identifier, use_templates=True):
        """Select the appropriate message template based on category."""
        if not use_templates:
            return self.get_detailed_input_msgs(image, cat, concept_identifier)
            
        person_categories = {'person', 'man', 'woman'}
        nature_categories = {"pet animal", "plant", "building", "cartoon character"}
        
        if cat in person_categories:
            return self.get_detailed_input_msgs_person(image, cat, concept_identifier)
        elif cat in nature_categories:
            return self.get_detailed_input_msgs(image, cat, concept_identifier)
        else:
            return self.get_detailed_input_msgs_household(image, cat, concept_identifier)

    def one_line_caption(self, image_file, cat, concept_identifier):
        """
        Generate a one-line caption for an image.
        
        Args:
            image_file: Path to image file
            cat: Category of object
            concept_identifier: Concept identifier tag
            
        Returns:
            str: One-line caption
        """
        image_1 = self.load_image('example_database/ydi.jpg')
        image_test = self.load_image(image_file)
        
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_1},
                    {"type": "text", "text": 'Describe the object in the image (in one sentence) which is identified by the concept identifier <ydi>. Make sure the caption starts with ydi.'}
                ]
            },
            {
                "role": "assistant",
                "content": "ydi is a yellow-white tote bag lying on the table."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_test},
                    {"type": "text", "text": f'Describe the {cat} in the image (in one sentence) which is identified by the concept identifier <{concept_identifier}>. Make sure the caption starts with {concept_identifier}.'}
                ]
            }
        ]
        
        output_text = self._generate_with_qwen(msgs, max_new_tokens=128)
        
        return output_text

import re
def extract_answer_term(text, term):
    patterns = {
        'info': r'"info":\s*(?:"([^"]+)"|([\w-]+))',
        'general': r'"general":\s*(?:"([^"]+)"|([\w-]+))',
        'category': r'"category":\s*(?:"([^"]+)"|([\w.]+))',
        'distinct features': r'"distinct features":\s*(?:"([^"]+)"|([\w.]+))',
    }
    
    pattern = patterns.get(term)
    if not pattern:
        return None
    
    match = re.search(pattern, text)
    if match:
        return match.group(1) or match.group(2)
    else:
        parts = text.split(term)
        if parts:
            return re.sub(r'[^a-zA-Z0-9\s]', '', parts[-1]).strip()
        return None

def build_entry(info_str, name, image_path):
    info_str = info_str.replace('```json\n', '').replace('\n```', '')

    try:
        info = json.loads(info_str)
        desc = info
        if 'category' in info:
            category = info['category']
        else:
            category = ''
        entry = {
            "name": name,
            "image": [image_path],
            "info": desc,
            "category": category,
        }
        return entry
    except Exception:
        print("Resolving error")
        entry = {
            "name": name,
            "image": [image_path],
            "info": {
                "general": extract_answer_term(info_str, "general"),
                "category": extract_answer_term(info_str, "category"),
                "distinct features": extract_answer_term(info_str, 'distinct features')
            },
            "category": extract_answer_term(info_str, "category"),
        }
    return entry

def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using Qwen2-VL model.")
    # parser.add_argument('--concept_name', type=str, required=True, help='The concept identifier for the object in the image.')
    parser.add_argument('--category_name', type=str, default='all', help='The category of the object in the image.')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-VL-7B-Instruct', help='Path to Qwen2-VL model')
    parser.add_argument('--template_based', action='store_true', help='Use template-based generation')
    parser.add_argument('--dataset', type=str, default="YoLLaVA", help='Specify the dataset source if needed.')
    parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility.')
    args = parser.parse_args()

    # Initialize Qwen2-VL descriptor
    qwen_descriptor = Qwen2VLDescription(model_path=args.model_path)
    
    # Build path to images
    path_to_image_files = os.path.join(
        f'example_database/{args.dataset}_seed_{args.seed}', 
        f'{args.category_name}', 
        # f'{args.concept_name}'
    )
    # import pdb;pdb.set_trace()
    # Find all images
    image_files = (
        glob.glob(os.path.join(path_to_image_files, '*.jpg')) + 
        glob.glob(os.path.join(path_to_image_files, '*.png'))
    )
    
    # Map category names
    category_mapping = {
        'tro_bag': 'trolley bag',
        'telephone': 'cell phone',
        'tumbler': 'glass',
        'veg': 'vegetable'
    }
    cat = category_mapping.get(args.category_name, args.category_name)
    
    # Generate captions
    db_entries = {"concept_dict": {}, "path_to_concept": {}}
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        concept_name = image_name.split('.')[0]
        key = f'<{concept_name}>'
        caption = qwen_descriptor.generate_caption(image_file, cat, concept_name, args)
        info_str = caption.replace('```json\n', '').replace('\n```', '')
        db_entries["concept_dict"][key] = build_entry(caption, concept_name, image_file)
        print(f"Caption for {image_file}: {caption}")
        # result_dict[concept_name] = [caption]
        db_entries["path_to_concept"][image_file] = key
    # Save results
    output_path = os.path.join(path_to_image_files, 'captions_qwen.json')
    with open(output_path, 'w') as f:
        json.dump(db_entries, f, indent=2)
    
    print(f"\nCaptions saved to: {output_path}")


if __name__ == '__main__':
    main()