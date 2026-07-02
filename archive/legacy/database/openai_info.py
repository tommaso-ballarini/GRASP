import base64
import os
from typing import Dict, Any
from pydantic import BaseModel
# Suppose you have an OpenAI Python client
import openai
from openai import OpenAI

# Initialize OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=openai.api_key)

def get_model_name_from_path(path: str) -> str:
    """Example function to extract a model name from the path."""
    return os.path.basename(path)

class ObjectDescription(BaseModel):
    description: str
    category: str

def get_image_info(
    image_path: str,
    concept_identifier: str,
) -> Dict[str, str]:
    """
    Encode the image in Base64, then pass that data plus the instructions to GPT.
    GPT typically cannot 'see' images unless you have GPT-4V access, but this code
    shows how you'd embed the image data in the prompt for experiments or local usage.
    
    Returns a dictionary with 'description' and 'category' keys, per your prompt instructions.
    """
    print(f"Getting info for image: {image_path}")
    # Read and encode the image to Base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create the prompt. We include the image data as text to show how one might pass it.
    # This is not guaranteed to work unless you have a model that can decode base64 images,
    # but demonstrates the approach.
    prompt = (
    "You are given an image in Base64-encoded form.\n"
    "Write a brief description of the object in the image and also guess the object category name.\n"
    "If the image is of an object focus on the size, color, texture, and shape of the object.\n"
    "If the image is of a person describe the person.\n"
    "Do not output any other information such as location, background, or setting.\n"
    f"The object in the image is identified as <{concept_identifier}>.\n"
    "You should return a dictionary with the following keys: 'description', 'category'.\n"
    "Example: {'description': '<{concept_identifier}> is a red apple with orange specks and a short stem', 'category': 'fruit'}.\n"
    # f"If you cannot determine the category precisely, use the metadata category provided for this image: {category}.\n"
    "The object category name is crucial because it will be passed to a YOLO detection module to obtain the bounding box of the object in the image.\n"
    )   

    try:
        # Call the model (assuming you have a custom model or GPT-4) 
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # or your custom model "gpt-4o-mini" if it supports images
            response_format=ObjectDescription,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type":"text",
                            "text": prompt
                        },
                        {
                            "type":"image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        }
                    ],
                }
            ],
        )

        # Parse the string returned by GPT
        raw_response = response.choices[0].message.content.strip()
        # Optionally, attempt to parse the GPT response as a Python dictionary
        # (Requires the response to be valid Python dictionary syntax)
        try:
            # Evaluate the string as Python code
            result_dict = eval(raw_response)

            # Ensure it has the keys we need
            if not isinstance(result_dict, dict):
                raise ValueError("Model response is not a dictionary.")
            if "description" not in result_dict or "category" not in result_dict:
                raise ValueError("Dictionary missing required keys: 'description' or 'category'.")

            return result_dict

        except Exception:
            # If GPT’s response is not a valid dict, just return the raw text
            return {
                "description": "Parsing error",
                "category": "unknown",
                "raw_response": raw_response,
            }

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return {
            "error_detail": str(e)
        }

def eval_model(args):
    """
    Example function that calls 'get_image_info'
    and prints or returns the output.
    """
    result = get_image_info(
        image_path=args.image_file,
        concept_identifier=args.concept_identifier,
    )
    print("Result:", result)
    return result

def main():
    # Example usage
    args =  type('Args', (), {
        "image_file": "example_database/decoration/oox.jpg",
        "concept_identifier": "ksz",
        "category": "reatail"
    })()
    eval_model(args)

if __name__ == "__main__":
    main()