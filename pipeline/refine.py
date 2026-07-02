"""
Refinement module for R2P-GEN (FLUX Edition).
Contains the logic for interacting with the API servers (VLM and FLUX)
to recover failed images.

Escalation strategy (3 levels):
  - Attempt 1 (gentle):   clear natural language, attributes placed at the start
  - Attempt 2 (emphatic): stronger adjectives, explicit position/color description
  - Attempt 3 (maximum):  prompt almost entirely focused on the missing attributes,
                           described in full detail; scene reduced to minimum
"""
import os
import io
import json
import base64
import requests
from PIL import Image


# ---------------------------------------------------------------------------
# Attribute classification helpers
# ---------------------------------------------------------------------------

# Maps fingerprint source key → attribute category for strategy selection
_KEY_TO_CATEGORY = {
    "brand/text": "logo_text",
    "pattern":    "pattern",
    "color":      "color_material",
    "material":   "color_material",
    "shape":      "general",
    "distinct features": "general",
}


def _classify_missing_attributes(missing_details: list, fingerprints: dict) -> dict:
    """
    For each missing detail string, determine its source fingerprint key category.

    Strategy:
      1. Build a reverse map: fingerprint_value → fingerprint_key
      2. Look up each missing detail in that map
      3. Translate the source key to a category

    Returns
    -------
    dict  {detail_string: category_string}
          category is one of: 'logo_text', 'pattern', 'color_material', 'general'
    """
    value_to_key: dict = {}
    if fingerprints:
        for k, v in fingerprints.items():
            if v and isinstance(v, str):
                value_to_key[v.strip()] = k

    result = {}
    for detail in missing_details:
        source_key = value_to_key.get(detail.strip())
        result[detail] = _KEY_TO_CATEGORY.get(source_key, "general")

    return result


def _build_attribute_instruction(detail: str, category: str, attempt: int) -> str:
    """
    Returns a single natural-language instruction for one missing attribute,
    calibrated to the current attempt level (1=gentle, 2=emphatic, 3=maximum).
    """
    if category == "logo_text":
        if attempt == 1:
            return (
                f"ensure the text or logo '{detail}' is clearly legible "
                f"and prominently visible on the object"
            )
        elif attempt == 2:
            return (
                f"make the text or logo '{detail}' the most prominent readable element "
                f"— describe its exact color, style, and position on the object"
            )
        else:
            return (
                f"describe explicitly the text or logo '{detail}': "
                f"its exact wording, color, font style, and placement "
                f"— this is the single defining feature of the object"
            )

    elif category == "pattern":
        if attempt == 1:
            return (
                f"ensure the '{detail}' is clearly visible "
                f"with its geometric structure and repetition intact"
            )
        elif attempt == 2:
            return (
                f"make the '{detail}' the dominant visual texture of the object "
                f"— describe its geometry, color, and repetition unit explicitly"
            )
        else:
            return (
                f"build the entire description around the '{detail}': "
                f"describe its geometry, scale, color, and repetition in full detail "
                f"— the pattern must be unmistakable"
            )

    elif category == "color_material":
        if attempt == 1:
            return (
                f"ensure '{detail}' is accurately rendered "
                f"and immediately recognizable as the object's primary characteristic"
            )
        elif attempt == 2:
            return (
                f"make '{detail}' unmistakably the primary visual characteristic "
                f"— place it at the very start of the description and reinforce it"
            )
        else:
            return (
                f"'{detail}' must be the absolute dominant characteristic "
                f"— state it explicitly at the beginning and reinforce it later "
                f"with a supporting phrase"
            )

    else:  # general
        if attempt == 1:
            return f"ensure '{detail}' is clearly visible"
        elif attempt == 2:
            return (
                f"prominently feature '{detail}' as a defining characteristic "
                f"of the object, described in explicit detail"
            )
        else:
            return (
                f"make '{detail}' the central focus of the description "
                f"— detail it explicitly and position it as the defining feature"
            )


def _build_rewrite_instruction(
    original_prompt: str,
    missing_details: list,
    attempt: int,
    history_context: str,
    fingerprints: dict,
) -> str:
    """
    Assembles the full rewrite instruction for Qwen3, using per-attribute
    natural-language guidance calibrated to the attempt level.
    No SDXL weight syntax is ever used.
    """
    classification = _classify_missing_attributes(missing_details, fingerprints or {})

    # Build per-attribute guidance lines
    attr_lines = "\n".join(
        f"  - {_build_attribute_instruction(d, classification[d], attempt)}"
        for d in missing_details
    )

    if attempt == 1:
        preamble = (
            "This image was generated but failed verification. "
            "Rewrite the prompt to fix the missing attributes listed below.\n"
            "Place the corrected attributes at the BEGINNING of the prompt "
            "using clear, descriptive language. "
            "Keep the scene, object category, background, and framing exactly the same."
        )
        closing = (
            "Use natural descriptive language only — no special syntax or weight notation. "
            "The rewritten prompt must read as a single fluent paragraph."
        )

    elif attempt == 2:
        preamble = (
            "This image still fails to show the required attributes. "
            "Rewrite the prompt with stronger, more explicit language for each missing attribute.\n"
            "Each missing attribute must appear at the very start of the prompt "
            "with explicit adjectives (e.g. 'prominently featuring', 'clearly showing', "
            "'with a distinctly visible'). "
            "Keep the background and scene context, but reduce generic scene description."
        )
        closing = (
            "Use natural descriptive language only — no special syntax or weight notation. "
            "The rewritten prompt must read as a single fluent paragraph."
        )

    else:  # attempt 3
        preamble = (
            "This image has repeatedly failed to show the required attributes. "
            "The prompt must now be almost entirely focused on the missing attributes.\n"
            "Reduce the scene description to the absolute minimum needed to identify "
            "the object category and background. "
            "Every remaining token must reinforce the missing attributes."
        )
        closing = (
            "Use natural descriptive language only — no special syntax or weight notation. "
            "The rewritten prompt must read as a single fluent paragraph. "
            "The missing attributes must be the first and last things mentioned."
        )

    instruction = (
        f"{preamble}\n\n"
        f"Missing attributes to fix:\n{attr_lines}\n\n"
        f"{history_context}"
        f"Original prompt: {original_prompt}\n\n"
        f"{closing}\n\n"
        f'Respond ONLY with JSON: {{"rewritten_prompt": "..."}}'
    )

    return instruction


# ---------------------------------------------------------------------------
# Main rewrite function
# ---------------------------------------------------------------------------

def qwen3_rewrite_prompt(
    reasoner,
    original_prompt: str,
    missing_details: list,
    attempt: int,
    failed_image_path: str = None,
    attempts_history: list = None,
    fingerprints: dict = None,
) -> str:
    """
    Rewrites the FLUX prompt using Qwen3-VL in multimodal mode when the
    failed image is available, or text-only as fallback.

    Parameters
    ----------
    reasoner          : Qwen3VLReasoning instance
    original_prompt   : current prompt that produced the failed image
    missing_details   : list of attribute strings that failed verification
    attempt           : current attempt number (1, 2, or 3)
    failed_image_path : path to the most recent failed generated image
                        (attempt N-1 image for attempt N;
                         _generated.png for attempt 1)
    attempts_history  : list of previous attempt summaries (for context)
    fingerprints      : fingerprints dict from the database entry
                        (used to classify attribute types)

    Returns
    -------
    str : rewritten prompt, or original_prompt on failure
    """
    # --- Build history context string ---
    history_context = ""
    if attempts_history:
        history_context = "Previous attempts summary:\n"
        for h in attempts_history:
            status = "improved" if h["improved"] else "no improvement"
            history_context += (
                f"  - Attempt {h['attempt']}: "
                f"missing {h['missing_before']} → after: {h['missing_after']} [{status}]\n"
            )
        history_context += "\n"

    instruction = _build_rewrite_instruction(
        original_prompt=original_prompt,
        missing_details=missing_details,
        attempt=attempt,
        history_context=history_context,
        fingerprints=fingerprints,
    )

    gen_text = None

    # --- Multimodal path (failed image available) ---
    if failed_image_path and os.path.exists(failed_image_path):
        try:
            failed_image = Image.open(failed_image_path).convert("RGB")
            msgs = reasoner.adapter.format_text_options_msgs(failed_image, instruction)
            _, gen_text = reasoner.model_interface.chat(msgs)
        except Exception as e:
            print(f"  [WARN] Multimodal rewrite failed, falling back to text-only: {e}")
            gen_text = None

    # --- Text-only fallback ---
    if gen_text is None:
        sys_msg = (
            "You are an expert AI prompt engineer for image diffusion models. "
            "Your task is to rewrite image generation prompts to recover missing "
            "physical details of an object. "
            "Use only natural descriptive language — never use weight syntax like (term:1.4). "
            'Output ONLY valid JSON: {"rewritten_prompt": "..."}'
        )
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": instruction},
        ]
        _, gen_text = reasoner.model_interface.chat(msgs)

    # --- Parse JSON response ---
    try:
        clean_text = gen_text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed = json.loads(clean_text)
        return parsed.get("rewritten_prompt", original_prompt)
    except Exception as e:
        print(f"  [WARN] JSON parse failed: {e}\nRaw: {gen_text}")
        return original_prompt


# ---------------------------------------------------------------------------
# Single-image HTTP generation (legacy, kept for compatibility)
# ---------------------------------------------------------------------------

def generate_recovery_http(
    flux_url: str,
    source_image_path: str,
    prompt: str,
    seed: int,
    output_path: str,
    max_retries: int = 2,
) -> bool:
    """
    Sends a single image + prompt to the FLUX HTTP server.
    Checks /health before each attempt.
    """
    if not os.path.exists(source_image_path):
        print(f"  ❌ Source image not found: {source_image_path}")
        return False

    try:
        img = Image.open(source_image_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"  ❌ Error reading source image: {e}")
        return False

    payload = {
        "prompts": [prompt],
        "seeds":   [seed],
        "source_image_b64": [b64_string],
    }

    for retry in range(max_retries + 1):
        try:
            health = requests.get(f"{flux_url}/health", timeout=5)
            if health.status_code != 200:
                raise ConnectionError("Server not healthy")
        except Exception as e:
            print(f"  ⚠️ FLUX unreachable (retry {retry}/{max_retries}): {e}")
            if retry < max_retries:
                import time
                time.sleep(10)
                continue
            print("  ❌ FLUX unreachable after all retries.")
            return False

        try:
            response = requests.post(f"{flux_url}/generate", json=payload, timeout=360)
            response.raise_for_status()
            result = response.json()

            error_msg = result.get("errors", [""])[0]
            if error_msg:
                print(f"  ❌ FLUX internal error: {error_msg}")
                return False

            img_b64 = result.get("images_b64", [""])[0]
            if not img_b64:
                print("  ❌ FLUX returned empty image.")
                return False

            img_bytes = base64.b64decode(img_b64)
            recovered_img = Image.open(io.BytesIO(img_bytes))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            recovered_img.save(output_path)
            return True

        except requests.Timeout:
            print(f"  ⚠️ FLUX timeout (retry {retry}/{max_retries}).")
            if retry < max_retries:
                import time
                time.sleep(5)
                continue
            print("  ❌ FLUX timeout after all retries.")
            return False

        except Exception as e:
            print(f"  ❌ FLUX API error: {e}")
            return False

    return False


# ---------------------------------------------------------------------------
# Batch HTTP generation
# ---------------------------------------------------------------------------

def _generate_batch_http(
    flux_url: str,
    source_image_paths: list,
    prompts: list,
    seeds: list,
    output_paths: list,
    max_retries: int = 2,
) -> list:
    """
    Sends a batch of images+prompts to the FLUX HTTP server in one call.
    Returns a list of bool (True = success) for each batch element.
    """
    results = [False] * len(prompts)

    # Encode source images
    sources_b64 = []
    for path in source_image_paths:
        try:
            img = Image.open(path).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            sources_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
        except Exception as e:
            print(f"  ❌ Error reading {path}: {e}")
            sources_b64.append(None)

    sources_b64_clean = [b or "" for b in sources_b64]

    payload = {
        "prompts":         prompts,
        "seeds":           seeds,
        "source_image_b64": sources_b64_clean,
    }

    for retry in range(max_retries + 1):
        try:
            health = requests.get(f"{flux_url}/health", timeout=5)
            if health.status_code != 200:
                raise ConnectionError("Server not healthy")
        except Exception as e:
            print(f"  ⚠️ FLUX unreachable (retry {retry}/{max_retries}): {e}")
            if retry < max_retries:
                import time
                time.sleep(10)
                continue
            return results

        try:
            response = requests.post(
                f"{flux_url}/generate", json=payload, timeout=360
            )
            response.raise_for_status()
            result = response.json()

            images_b64 = result.get("images_b64", [])
            errors     = result.get("errors", [""] * len(prompts))

            for i, (img_b64, err) in enumerate(zip(images_b64, errors)):
                if err:
                    print(f"  ❌ FLUX error item {i}: {err}")
                    continue
                if not img_b64:
                    print(f"  ❌ FLUX item {i}: empty image.")
                    continue
                try:
                    img_bytes = base64.b64decode(img_b64)
                    img = Image.open(io.BytesIO(img_bytes))
                    os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)
                    img.save(output_paths[i])
                    results[i] = True
                except Exception as e:
                    print(f"  ❌ Error saving item {i}: {e}")

            return results

        except requests.Timeout:
            print(f"  ⚠️ FLUX timeout (retry {retry}/{max_retries}).")
            if retry < max_retries:
                import time
                time.sleep(5)
                continue
            return results

        except Exception as e:
            print(f"  ❌ FLUX API error: {e}")
            return results

    return results