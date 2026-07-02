import re
import os
import torch
import glob
from PIL import Image
import requests
from io import BytesIO
import json
import sys
from filelock import FileLock
import argparse
from transformers import CLIPModel, CLIPProcessor
from transformers import AutoModel, AutoTokenizer
from defined import categories
from compute_confidence import ClipScoreCalculator


# --- FIX IMPORT ROBUSTO ---
# Calcola il percorso assoluto della cartella corrente (database)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Risali alla cartella 'r2p_core' (che contiene database, utils, evaluators)
core_dir = os.path.dirname(current_dir)

# Costruisci i percorsi assoluti per le cartelle dipendenti
utils_path = os.path.join(core_dir, "utils")
evaluators_path = os.path.join(core_dir, "evaluators")
models_path = os.path.join(core_dir, "models")  # Aggiungo anche models per sicurezza


if utils_path not in sys.path:
    sys.path.append(utils_path)
if evaluators_path not in sys.path:
    sys.path.append(evaluators_path)
if models_path not in sys.path:
    sys.path.append(models_path)


def clean_and_load_json(model_output):
    if not model_output.strip():
        raise ValueError("Received empty model output!")

    # Remove markdown code block markers (` ```json ... ``` `)
    model_output = model_output.strip("```json").strip("```").strip()

    # Extract only JSON part if there's extra text
    clean_json_match = re.search(r'\{.*\}', model_output, re.DOTALL)
    if clean_json_match:
        cleaned_json = clean_json_match.group(0)
    else:
        raise ValueError(f"Invalid JSON format: {model_output}")

    return cleaned_json

class MiniCPMDescription:
    def __init__(self, 
                model_path="openbmb/MiniCPM-o-2_6",
                device="cuda",
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16):
        """
        Initializes the MiniCPM model for generating object descriptions.

        Args:
            model_path (str): Hugging Face model path.
            device (str): Device to load the model on (e.g., 'cuda' or 'cpu').
            attn_implementation (str): Attention mechanism ('sdpa' or 'flash_attention_2').
            torch_dtype (dtype): Torch data type (default: bfloat16).
        """
        print("Loading Description Generator")
        self.device = device
        model_path = 'openbmb/MiniCPM-o-2_6'
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True,
            attn_implementation=attn_implementation, 
            torch_dtype=torch_dtype
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336').to(self.device)
        self.feature_extractor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.clip_scorer = ClipScoreCalculator(self.feature_extractor, self.clip_model)

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
            '"distinct features": "[Silver hair, glasses, confident expression]"'
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
                {'role': 'user', 'content': [image_1, question_example]},
                {'role': 'assistant', 'content': answer_example},
                {'role': 'user', 'content': [image_test, question_test]}
            ]
        else:
            # Zero-shot fallback
            msgs = [
                {'role': 'user', 'content': [image_test, question_test]}
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
            '"brand/text": ""No visible text or brand markings.", '
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
            {'role': 'user', 'content': [image_1, question_example]},
            {'role': 'assistant', 'content': answer_example},
            {'role': 'user', 'content': [image_test, question_test]}
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
            '"distinct features": "[Upper half is white, lower half is yellow., Strong contrast between the two sections, "Amazon style printed at the center"]", '
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
            {'role': 'user', 'content': [image_1, question_example]},
            {'role': 'assistant', 'content': answer_example},
            {'role': 'user', 'content': [image_test, question_test]}
        ]
        return msgs

    def get_input_msgs_generic(self, image_test, cat, concept_identifier):
        image_example = self.load_image('example_database/bo.png')
        answer_format = {"info": "<a brief description of the object>", "category": "<category of the object in the image>"}
        msgs = [
            {
                'role': 'user', 
                'content': [
                    image_example, 
                    f"""Describe the dog in the image that is identified by the concept-identifier, <bo>.
                    Your response must follow EXACTLY the JSON format below:
                    {json.dumps(answer_format)}
                    IMPORTANT: Your "info" field MUST begin with the <concept-identifier>. For example:
                    "<bo> is a ..."
                    Any deviation from this format will be considered incorrect.
                    """
                ]
            },
            {
                'role':'assistant', 
                'content':'{"info": "<bo> is a well-groomed, medium-sized Shiba Inu with a thick, cinnamon-colored coat, cream accents, alert eyes, and a black collar.", "category": "dog"}'
            },
            {
                'role': 'user', 
                'content': [
                    image_test, 
                    f"""Now, describe the {cat} in the image identified by the concept-identifier, <{concept_identifier}>.
                    Your response must follow EXACTLY the JSON format below:
                    {json.dumps(answer_format)}
                    IMPORTANT: The description in the "info" field MUST begin with "<{concept_identifier}> is ...".
                    Any deviation from this format will be considered incorrect.
                     """
                ]
            }
        ]
        
        return msgs

    def generate_caption(self, image_file, cat, concept_identifier, args):

        # Load image
        image = self.load_image(image_file)
        
        # Auto-detect category if not provided
        cat = self._determine_category(image, cat)
        
        # Select appropriate message template
        msgs = self._select_template(image, cat, concept_identifier, args.template_based)
        
        # Generate and return caption
        with torch.inference_mode():
            # MODIFICA QUI: Gestiamo il ritorno variabile di MiniCPM-o-2.6
            res = self.model.chat(msgs=msgs, tokenizer=self.tokenizer)
            
            # Se ritorna una tupla (response, history, ...), prendiamo il primo elemento
            if isinstance(res, (tuple, list)):
                output_text = res[0]
            else:
                output_text = res
        
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
        similarities, _ = self.clip_scorer.clip_score(image, prompts)
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
        image_1 = self.load_image('example_database/method_mean_feat_swapped/bag/ydi_0.jpg')
        image_test = self.load_image(image_file)
        msgs = [{'role':'user', 'content':[image_1, 'Describe the object in the image (in one sentence) which is identified by the concept identifier <ydi>. Make sure the caption starts with ydi.']},
                {'role':'assistant', 'content':"ydi is a yellow-white tote bag lying on the table."},
                {'role':'user', 'content':[image_test, f'Describe the {cat} in the image (in one sentence) which is identified by the concept identifier <{concept_identifier}>. Make sure the caption starts with {concept_identifier}.']}
                ]
        with torch.inference_mode():
            output_text = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer
            )
        return output_text

def main():
    parser = argparse.ArgumentParser(description="Generate captions for images using MiniCPM model.")
    parser.add_argument('--concept_name', type=str, required=True, help='The concept identifier for the object in the image.')
    parser.add_argument('--category_name', type=str, required=True, help='The category of the object in the image.')
    args = parser.parse_args()

    mini_cpm = MiniCPMDescription()
    path_to_image_files = os.path.join('../YoLLaVA/yollava-data/train_', 
                                       f'{args.category_name}', 
                                       f'{args.concept_name}')
    image_files = glob.glob(os.path.join(path_to_image_files, '*.jpg')) + glob.glob(os.path.join(path_to_image_files, '*.png'))
    if args.category_name == 'tro_bag':
        cat = 'trolley bag'
    elif args.category_name == 'telephone':
        cat = 'cell phone'
    elif args.category_name == 'tumbler':
        cat = 'glass'
    elif args.category_name == 'veg':
        cat = 'vegetable'
    else:
        cat = args.category_name
    result_dict = {}
    for image_file in image_files:
        image_name = os.path.basename(image_file)
        caption = mini_cpm.one_line_caption(image_file, cat, args.concept_name)
        print(f"Caption for {image_file}: {caption}")
        result_dict[image_name] = [caption]
    
    with open(os.path.join(path_to_image_files, 'captions.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)

    
if __name__ == '__main__':
    main()
