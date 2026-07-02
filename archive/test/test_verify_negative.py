import sys, os, json
sys.path.insert(0, '/leonardo/home/userexternal/tballari/R2P-GEN')

os.environ['R2P_MODELS_BASE'] = '/leonardo_work/IscrC_MUSE/tballari/models_cache/huggingface'
os.environ['R2P_CLUSTER_MODE'] = 'true'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from config import Config
from r2p_core.models.qwen3_vl_reasoning import Qwen3VLReasoning
from pipeline.r2p_tools import ClipScoreCalculator
from pipeline.verify import verify_generation_r2p

OUTPUT_DIR = '/leonardo_work/IscrC_MUSE/tballari/FM_Data/output/test_debug'
DB_PATH = '/leonardo/home/userexternal/tballari/R2P-GEN/database/database.json'

with open(DB_PATH) as f:
    db = json.load(f)
concepts = db['concept_dict']

print("Caricamento modelli...")
reasoner = Qwen3VLReasoning(model_path=Config.Models.QWEN3_MODEL, device="cuda")
clip_calculator = ClipScoreCalculator(device="cuda")

# Prende i primi due concetti
ids = list(concepts.keys())[:2]
c0, c1 = ids[0], ids[1]

gen_c0 = f"{OUTPUT_DIR}/{c0}_generated.png"
gen_c1 = f"{OUTPUT_DIR}/{c1}_generated.png"
ref_c0 = concepts[c0]['representative_image']
ref_c1 = concepts[c1]['representative_image']
info_c0 = concepts[c0]['info']
info_c1 = concepts[c1]['info']

print(f"\n{'='*60}")
print(f"TEST POSITIVO: {c0} generata vs {c0} reference (dovrebbe PASSARE)")
print(f"{'='*60}")
result = verify_generation_r2p(reasoner, clip_calculator, gen_c0, ref_c0, info_c0)
print(f"Risultato: {'✅ PASS' if result['is_verified'] else '❌ FAIL'} | score={result['score']:.3f} | method={result['method']}")

print(f"\n{'='*60}")
print(f"TEST NEGATIVO: {c1} generata vs {c0} reference (dovrebbe FALLIRE)")
print(f"{'='*60}")
result_neg = verify_generation_r2p(reasoner, clip_calculator, gen_c1, ref_c0, info_c0)
print(f"Risultato: {'✅ PASS' if result_neg['is_verified'] else '❌ FAIL'} | score={result_neg['score']:.3f} | method={result_neg['method']}")
