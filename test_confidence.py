import sys
sys.path.insert(0, '/leonardo/home/userexternal/tballari/R2P-GEN')

import os
os.environ['R2P_MODELS_BASE'] = '/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface'
os.environ['R2P_CLUSTER_MODE'] = 'true'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from PIL import Image
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
from config import Config

reasoner = Qwen3VLReasoning(
    model_path=Config.Models.QWEN3_MODEL,
    device="cuda",
)

img = Image.open('/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_debug/<alx>_generated.png').convert('RGB')

prompt = (
    "Look at the image. Is the feature 'leather texture' clearly visible and correct?\n"
    'Respond only with JSON: {"answer": "yes"} or {"answer": "no"}.'
)

msgs = reasoner.adapter.format_text_options_msgs(img, prompt)
outputs, text = reasoner.model_interface.chat(msgs)

print(f"\n🔍 Testo generato: {text}")
print(f"🔍 Tipo sequences: {type(outputs['sequences'])}")
print(f"🔍 Shape sequences: {outputs['sequences'].shape}")

tokens = reasoner.conf_calculator.tokenizer.convert_ids_to_tokens(outputs['sequences'][0])
print(f"🔍 Token generati: {tokens}")

confidence = reasoner.conf_calculator.calculate_confidence(outputs, task="recognition")
print(f"\n✅ Confidence: yes={confidence['yes_confidence']:.3f} | no={confidence['no_confidence']:.3f}")
