"""
Modulo di Recovery per R2P-GEN (FLUX Edition).
Contiene le logiche per l'interazione con i server API (VLM e FLUX) 
per tentare il recupero delle immagini fallite.
"""
import os
import io
import json
import base64
import requests
from PIL import Image


def qwen3_rewrite_prompt(reasoner, original_prompt: str, missing_details: list, attempt: int) -> str:
    """Riscrive il prompt usando l'oggetto Qwen3VLReasoning (testo-solo) nativo in PyTorch."""
    
    sys_msg = (
        "You are an expert AI prompt engineer for image diffusion models. "
        "Your task is to rewrite the prompt to recover missing physical details of an object. "
        "Output ONLY valid JSON in this format: {\"rewritten_prompt\": \"...\"}"
    )
    missing_str = ", ".join(missing_details)
    
    if attempt <= 2:
        user_msg = (
            f"The image generated from the prompt missed these critical object details: {missing_str}.\n"
            f"Rewrite the prompt to give strong discursive emphasis to these features.\n"
            f"Use adjectives like 'highly detailed', 'prominent', or 'clearly visible' right before the missing attributes.\n"
            f"CRITICAL: Keep the overall scene, object category, and framing EXACTLY the same.\n"
            f"Original prompt: {original_prompt}"
        )
    else:
        user_msg = (
            f"The image has REPEATEDLY missed these critical details: {missing_str}.\n"
            f"Rewrite the prompt to forcefully anchor the text-encoder:\n"
            f"1) Write the missing attributes in ALL CAPS (UPPERCASE).\n"
            f"2) Repeat the missing attributes twice if necessary to force attention.\n"
            f"3) Example: 'a prominent WHITE LOGO, clearly featuring the WHITE LOGO'.\n"
            f"CRITICAL: Do NOT change the scene context, background, or framing.\n"
            f"Original prompt: {original_prompt}"
        )

    # Creiamo i messaggi nel formato standard chat
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]
    
    try:
        # ✨ CHIAMATA NATIVA AL TUO OGGETTO REASONER ✨
        outputs, gen_text = reasoner.model_interface.chat(messages)
        
        clean_text = gen_text.strip().removeprefix('```json').removesuffix('```').strip()
        parsed = json.loads(clean_text)
        return parsed.get('rewritten_prompt', original_prompt)
        
    except Exception as e:
        print(f"  [WARN] Fallito parsing JSON o inferenza: {e}\nRaw output: {gen_text if 'gen_text' in locals() else 'N/A'}")
        return original_prompt


def generate_recovery_http(flux_url: str, source_image_path: str, prompt: str, seed: int, output_path: str) -> bool:
    """
    Invia l'immagine in Base64 e il nuovo prompt al server FLUX.
    """
    if not os.path.exists(source_image_path):
        print(f"  ❌ Immagine sorgente non trovata: {source_image_path}")
        return False

    try:
        img = Image.open(source_image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64_string = base64.b64encode(buf.getvalue()).decode('utf-8')

        payload = {
            "prompts": [prompt],
            "seeds": [seed],
            "source_image_b64": [b64_string]
        }

        response = requests.post(f"{flux_url}/generate", json=payload, timeout=360)
        response.raise_for_status()
        result = response.json()

        error_msg = result.get('errors', [""])[0]
        if error_msg:
            print(f"  ❌ Errore interno server FLUX: {error_msg}")
            return False
            
        img_b64_response = result.get('images_b64', [""])[0]
        if img_b64_response:
            img_bytes = base64.b64decode(img_b64_response)
            recovered_img = Image.open(io.BytesIO(img_bytes))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            recovered_img.save(output_path)
            return True
        else:
            print("  ❌ FLUX ha restituito un'immagine vuota.")
            return False

    except Exception as e:
        print(f"  ❌ Errore API FLUX: {e}")
        return False