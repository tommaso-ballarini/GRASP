import json
import os
import torch
from config import Config
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning

# --- CONFIGURAZIONE PATHS ---
OUTPUT_DIR = "/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_30"
REJECTED_FILE = os.path.join(OUTPUT_DIR, "rejected_concepts.json")
PROMPTS_FILE = os.path.join(OUTPUT_DIR, "prompts.json")
DRY_RUN_OUTPUT = os.path.join(OUTPUT_DIR, "refined_prompts_qwen3.json")

def qwen3_rewrite_prompt(reasoner, original_prompt: str, missing_details: list, attempt: int) -> str:
    """Riscrive il prompt usando l'oggetto Qwen3VLReasoning (testo-solo)."""
    
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
        # ✨ LA RIGA CORRETTA BASATA SUL TUO verify.py ✨
        outputs, gen_text = reasoner.model_interface.chat(messages)
        
        clean_text = gen_text.strip().removeprefix('```json').removesuffix('```').strip()
        parsed = json.loads(clean_text)
        return parsed.get('rewritten_prompt', original_prompt)
        
    except Exception as e:
        print(f"  [WARN] Fallito parsing JSON o inferenza: {e}\nRaw output: {gen_text if 'gen_text' in locals() else 'N/A'}")
        return original_prompt

def test_recovery_prompts():
    print(f"🚀 Avvio Dry Run Recovery usando Qwen3-VL...")
    
    if not os.path.exists(REJECTED_FILE) or not os.path.exists(PROMPTS_FILE):
        print("❌ File JSON mancanti.")
        return

    with open(REJECTED_FILE, 'r') as f:
        rejected_data = json.load(f)
    with open(PROMPTS_FILE, 'r') as f:
        prompts_data = json.load(f)

    print("Caricamento Qwen3-VL in VRAM...")
    reasoner = Qwen3VLReasoning(
        model_path=Config.Models.QWEN3_MODEL,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        seed=Config.Generate.SEED,
    )

    refined_results = {}

    for concept, rej_info in rejected_data.items():
        if concept not in prompts_data:
            continue
            
        original_prompt = prompts_data[concept]["flux_prompt"]
        failed_attributes = rej_info.get("details", {}).get("failed_attributes", [])
        if not failed_attributes:
            failed_attributes = rej_info.get("missing_details", [])

        print(f"\nAnalizzo concept: {concept}")
        print(f"  ❌ Missing: {failed_attributes}")

        print("  🔄 Attempt 1...")
        new_prompt_att1 = qwen3_rewrite_prompt(reasoner, original_prompt, failed_attributes, 1)
        
        print("  🔄 Attempt 3 (Hard)...")
        new_prompt_att3 = qwen3_rewrite_prompt(reasoner, original_prompt, failed_attributes, 3)

        refined_results[concept] = {
            "failed_attributes": failed_attributes,
            "original_prompt": original_prompt,
            "refined_prompt_attempt_1": new_prompt_att1,
            "refined_prompt_attempt_3": new_prompt_att3
        }

    with open(DRY_RUN_OUTPUT, 'w') as f:
        json.dump(refined_results, f, indent=4)
        
    print(f"\n✅ Dry run completata. Guarda il file: {DRY_RUN_OUTPUT}")

if __name__ == "__main__":
    test_recovery_prompts()