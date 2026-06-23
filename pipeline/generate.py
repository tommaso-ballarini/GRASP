import os
import json
import torch
import gc
import time
from PIL import Image
from tqdm import tqdm

# Importiamo le AutoPipelines per gestire in sicurezza T2I e I2I
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

from config import Config
from pipeline.prompts.flux_prompts import build_flux_prompt

class Generator:
    def __init__(self, database_path: str, output_dir: str, device: str = "cuda:0",
                 num_shards: int = 1, shard_index: int = 0,
                 use_naive_prompt: bool = False, no_image_cond: bool = False):
        """
        Inizializza il generatore.
        Aggiunti parametri per ablazioni: use_naive_prompt e no_image_cond.
        """
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.database_path = database_path
        self.output_dir = output_dir
        self.device = device
        
        # Flag per le ablazioni
        self.use_naive_prompt = use_naive_prompt
        self.no_image_cond = no_image_cond
        
        self.pipe = None
        self._load_pipeline()
        
    def _load_pipeline(self):
        """Carica FLUX (T2I o I2I a seconda delle ablazioni)."""
        print(f"\n🚀 Inizializzazione FLUX in corso...")
        model_name_or_path = getattr(Config.Models, "FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        
        # Scegliamo la pipeline corretta in base all'ablazione
        if self.no_image_cond:
            print("   ℹ️ Modalità Text-Only (no image conditioning) attivata.")
            pipe_class = AutoPipelineForText2Image
        else:
            print("   ℹ️ Modalità Image-to-Image attivata.")
            pipe_class = AutoPipelineForImage2Image

        self.pipe = pipe_class.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" 
        )
        self.pipe.to(self.device)   
        print("✅ FLUX caricato con successo.")

    def generate_all(self):
        num_shards = self.num_shards
        shard_index = self.shard_index
        with open(self.database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
            
        concept_dict = database.get("concept_dict", {})
        
        # LOGICA DI SHARDING
        all_keys = sorted(list(concept_dict.keys()))
        chunk_size = len(all_keys) // num_shards
        start_idx = shard_index * chunk_size
        end_idx = start_idx + chunk_size if shard_index < num_shards - 1 else len(all_keys)
        my_keys = all_keys[start_idx:end_idx]
        
        print(f"🖥️  Worker {shard_index}/{num_shards} - Assegnati {len(my_keys)} concetti su {len(all_keys)} totali.")
        stats = {"total": len(my_keys), "success": 0, "failed": 0}
        
        prompts_log = {}
        target_context = Config.get_background_template()
        
        t_start = time.time()
        for i, concept_id in enumerate(tqdm(my_keys, desc="Generating", unit="img")):
            content = concept_dict[concept_id]
            t_concept = time.time()
            print(f"\n🎨 [{i+1}/{len(my_keys)}] Elaborazione concetto: {concept_id}")
            try:
                # 1. Recupero dell'immagine sorgente (RAPPRESENTATIVA)
                source_image_path = content.get("representative_image")
                if not source_image_path:
                    # Fallback: usa la prima immagine della lista "image"
                    images = content.get("image", [])
                    if images:
                        source_image_path = images[0]
                    else:
                        print(f"   ⚠️ Nessuna immagine sorgente trovata per {concept_id}.")
                        continue
                # Rendi il percorso assoluto (per sicurezza)
                source_image_path = os.path.abspath(source_image_path)
                
                # 3. Costruzione del prompt FLUX (Ablazione Naive vs Fingerprints)
                if self.use_naive_prompt:
                    category = content.get("category", "object")
                    flux_prompt = f"A high-quality photograph of a {category}. The {category} is {target_context}."
                    print(f"   📝 NAIVE Prompt: {flux_prompt[:100]}...")
                else:
                    flux_prompt = build_flux_prompt(attributes, target_context)
                    print(f"   📝 FULL Prompt: {flux_prompt[:100]}...")
                
                # Setup generatore per riproducibilità
                generator = torch.Generator(device=self.device).manual_seed(Config.Generate.SEED)
                
                # 4 & 5. Generazione con FLUX (Ablazione Text-Only vs Img2Img)
                with torch.inference_mode():
                    if self.no_image_cond:
                        # Condizioni A, B: Niente immagine in input
                        output_img = self.pipe(
                            prompt=flux_prompt,
                            num_inference_steps=4,
                            generator=generator
                        ).images[0]
                    else:
                        # Condizioni C, D, E, F: Usa immagine
                        seed_img = Image.open(source_image_path).convert("RGB")
                        target_size = Config.Images.REFERENCE_IMAGE_SIZE
                        seed_img = seed_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                        
                        output_img = self.pipe(
                            prompt=flux_prompt,
                            image=seed_img,
                            num_inference_steps=4,
                            generator=generator
                        ).images[0]

                # # 2. Recupero degli attributi (Fingerprints)
                # attributes = content.get("info", {})
                
                # # 3. Costruzione del prompt FLUX
                # flux_prompt = build_flux_prompt(attributes, target_context)
                # print(f"   📝 Prompt: {flux_prompt[:100]}...")
                
                # # 4. Caricamento e standardizzazione immagine
                # seed_img = Image.open(source_image_path).convert("RGB")
                # target_size = Config.Images.REFERENCE_IMAGE_SIZE
                # seed_img = seed_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # generator = torch.Generator(device=self.device).manual_seed(Config.Generate.SEED)
                
                # # 5. Generazione con FLUX Img2Img
                # with torch.inference_mode():
                #     output_img = self.pipe(
                #         prompt=flux_prompt,
                #         image=seed_img,
                #         num_inference_steps=4,
                #         generator=generator
                #     ).images[0]
                
                # 6. Salvataggio Output (percorso assoluto)
                out_path = os.path.join(self.output_dir, f"{concept_id}_generated.png")
                out_path = os.path.abspath(out_path)
                output_img.save(out_path)
                
                prompts_log[concept_id] = {
                    "flux_prompt": flux_prompt,
                    "source_image": source_image_path,
                    "output_image": out_path,
                }
                
                elapsed = time.time() - t_concept
                total_elapsed = time.time() - t_start
                avg = total_elapsed / (i + 1)
                remaining = avg * (len(my_keys) - i - 1)
                stats["success"] += 1
                print(f"   ✅ Salvato: {os.path.basename(out_path)} "
                    f"| {elapsed:.1f}s | avg {avg:.1f}s/img | ETA {remaining/60:.1f}min")
            except Exception as e:
                print(f"   ❌ Errore {concept_id}: {str(e)}")
                stats["failed"] += 1
        
        # Salva il log
        prompts_path = os.path.join(self.output_dir, "prompts.json")
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(prompts_log, f, indent=4, ensure_ascii=False)
        print(f"   📝 Prompt log salvato → {prompts_path}")
        
        total = time.time() - t_start
        print(f"\n⏱️  Tempo totale: {total/60:.1f}min | "
            f"avg {total/max(stats['success'],1):.1f}s/img")
        return stats

    def cleanup(self):
        """Libera rigorosamente la memoria."""
        print("\n🧹 Pulizia VRAM da FLUX...")
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()