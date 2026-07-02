"""
pipeline/generate_dreambench.py

Generazione massiva Zero-Shot per il benchmark DreamBench.
Utilizza il server FLUX HTTP attivo per generare N immagini per ogni prompt
basandosi sulla "subject phrase" estratta da Qwen3-VL.
"""

import os
import json
import argparse
from tqdm import tqdm

from pipeline.prompts.dreambench_prompt_compiler import compile_subject_phrase, build_dreambench_prompt
from pipeline.prompts.dreambench_prompts import get_prompts_for_entity_type, is_property_modification_prompt
from pipeline.refine import _generate_batch_http
from config import Config

def _get_first_image(content: dict) -> str | None:
    """Recupera l'immagine di reference (migliore possibile) dal concept."""
    if content.get("representative_image"):
        return content["representative_image"]
    top_k = content.get("top_k_images")
    if top_k:
        return top_k[0]
    value = content.get("image")
    if isinstance(value, list) and value:
        return value[0]
    return None

def _sanitize_folder_name(name: str) -> str:
    """Rimuove i tag <> dal concept_id per creare cartelle pulite (es. <dog> -> dog)."""
    return name.strip("<>")

def run_dreambench_generation(database_path: str, output_dir: str, num_images_per_prompt: int, batch_size: int):
    print(f"\n{'='*70}")
    print("🚀 AVVIO GENERAZIONE DREAMBENCH (ZERO-SHOT) TRAMITE FLUX HTTP")
    print(f"{'='*70}")
    
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database non trovato: {database_path}")

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
        
    concept_dict = database.get("concept_dict", {})
    flux_url = getattr(Config.Models, "RECOVERY_FLUX_URL", "http://127.0.0.1:8766")
    
    print(f"📦 Concetti da elaborare: {len(concept_dict)}")
    print(f"🔗 Endpoint FLUX: {flux_url}")
    print(f"📁 Output Directory: {output_dir}\n")

    # 1. Prepariamo la coda totale delle richieste
    requests_queue = []
    
    for concept_id, content in concept_dict.items():
        fingerprints = content.get("info", {})
        entity_type = fingerprints.get("_entity_type", "OBJECT")
        
        # Generiamo la stringa naturale del soggetto
        subject_phrase = compile_subject_phrase(fingerprints)
        
        # Recuperiamo l'immagine di reference per SDEdit
        ref_image_path = _get_first_image(content)
        if not ref_image_path or not os.path.exists(ref_image_path):
            print(f"⚠️ Salto {concept_id}: immagine di reference mancante o non trovata.")
            continue
            
        # Recuperiamo i prompt ufficiali (25 per classe)
        prompts = get_prompts_for_entity_type(entity_type)
        clean_concept_name = _sanitize_folder_name(concept_id)
        
        for prompt_idx, prompt_template in enumerate(prompts):
            # I prompt 20-24 sono property modification: il modificatore
            # ("red", "cube shaped"...) deve restare attaccato al soggetto,
            # quindi NON applichiamo la separazione fotografica.
            is_prop_mod = is_property_modification_prompt(prompt_idx)
            final_prompt = build_dreambench_prompt(
                prompt_template,
                subject_phrase,
                is_property_modification=is_prop_mod,
            )
            
            # Creiamo la directory di output per il prompt corrente
            # Struttura: output_dir/backpack_dog/00/
            prompt_dir = os.path.join(output_dir, clean_concept_name, f"{prompt_idx:02d}")
            os.makedirs(prompt_dir, exist_ok=True)
            
            for img_idx in range(num_images_per_prompt):
                out_path = os.path.join(prompt_dir, f"{img_idx}.png")
                
                # Se l'immagine esiste già, la saltiamo (utile per resume in caso di crash)
                if os.path.exists(out_path):
                    continue
                    
                # Distanziamo i seed per garantire varietà visiva nei batch
                seed = Config.Generate.SEED + (prompt_idx * 100) + img_idx
                
                requests_queue.append({
                    "concept_id": concept_id,
                    "prompt": final_prompt,
                    "source_image": ref_image_path,
                    "seed": seed,
                    "output_path": out_path
                })

    total_requests = len(requests_queue)
    if total_requests == 0:
        print("✅ Nessuna immagine da generare (tutte già esistenti o coda vuota).")
        return

    print(f"🎯 Coda pronta: {total_requests} immagini da generare.")
    print(f"   Batch size: {batch_size}")
    
    # 2. Suddividiamo la coda in batch e inviamo le richieste
    successful_gens = 0
    
    for i in tqdm(range(0, total_requests, batch_size), desc="Batch FLUX HTTP"):
        batch = requests_queue[i : i + batch_size]
        
        b_sources = [req["source_image"] for req in batch]
        b_prompts = [req["prompt"] for req in batch]
        b_seeds = [req["seed"] for req in batch]
        b_outputs = [req["output_path"] for req in batch]
        
        try:
            results = _generate_batch_http(
                flux_url=flux_url,
                source_image_paths=b_sources,
                prompts=b_prompts,
                seeds=b_seeds,
                output_paths=b_outputs
            )
            successful_gens += sum(1 for res in results if res is True)
        except Exception as e:
            print(f"\n❌ Errore critico nel batch {i//batch_size}: {e}")
            
    print(f"\n🏁 Generazione Completata! Successi: {successful_gens}/{total_requests}")
    print(f"📁 Le immagini sono disponibili in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generazione DreamBench R2P-GEN")
    parser.add_argument("--database", type=str, required=True, help="Path al database_db.json")
    parser.add_argument("--output", type=str, default="output_dreambench", help="Cartella root di output")
    parser.add_argument("--images-per-prompt", type=int, default=4, help="Numero di immagini per prompt (N)")
    parser.add_argument("--batch-size", type=int, default=4, help="Dimensione del batch per la richiesta HTTP")
    
    args = parser.parse_args()
    
    run_dreambench_generation(
        database_path=args.database,
        output_dir=args.output,
        num_images_per_prompt=args.images_per_prompt,
        batch_size=args.batch_size
    )