"""
FLUX-based Image Generator per R2P-GEN.
Sostituisce la vecchia pipeline SDXL + IP-Adapter.
Utilizza Textual Anchoring e SDEdit (rumore nativo img2img) per preservare l'identità.
"""

import os
import json
import torch
import gc
from PIL import Image
from diffusers import DiffusionPipeline
import diffusers.models.attention_processor
import torch.nn.functional as F

from config import Config
# Importiamo il nuovo costruttore di prompt
from pipeline.prompts.flux_prompts import build_flux_prompt

class Generator:
    def __init__(self, database_path: str, output_dir: str, device: str = "cuda:0"):
        """
        Inizializza il generatore mantenendo la compatibilità con full_loop.py.
        """
        self.database_path = database_path
        self.output_dir = output_dir
        self.device = device
        self.pipe = None
        
        self._load_pipeline()
        
    def _load_pipeline(self):
        """Carica FLUX ottimizzando al massimo l'uso della memoria."""
        print(f"\n🚀 Inizializzazione FLUX Img2Img in corso...")
        
        # Patch Diffusers per l'ottimizzazione della memoria (SDPA)
        _original_sdpa = F.scaled_dot_product_attention
        def _stripped_sdpa(*args, **kwargs):
            kwargs.pop('enable_gqa', None)
            return _original_sdpa(*args, **kwargs)
        diffusers.models.attention_processor.F.scaled_dot_product_attention = _stripped_sdpa

        # ATTENZIONE: Assicurati di aggiungere FLUX_MODEL nel tuo config.py
        # Es: Config.Models.FLUX_MODEL = "black-forest-labs/FLUX.1-schnell" o il path locale
        model_name_or_path = getattr(Config.Models, "FLUX_MODEL", "black-forest-labs/FLUX.1-schnell")
        
        self.pipe = DiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # CRUCIALE: Offload su CPU. Permette al modello di pesare meno in VRAM 
        # quando non genera, lasciando spazio a MiniCPM/Qwen per la verifica.
        self.pipe.enable_model_cpu_offload()
        print("✅ FLUX caricato con successo (CPU Offload attivato).")

    # Sostituisci la firma di generate_all con questa:
    def generate_all(self, num_shards: int = 1, shard_index: int = 0):
        with open(self.database_path, 'r', encoding='utf-8') as f:
            database = json.load(f)
            
        concept_dict = database.get("concept_dict", {})
        
        # LOGICA DI SHARDING
        # 1. Estraiamo le chiavi e le ordiniamo (fondamentale per la coerenza tra processi!)
        all_keys = sorted(list(concept_dict.keys()))
        
        # 2. Calcoliamo la porzione spettante a questa GPU
        chunk_size = len(all_keys) // num_shards
        start_idx = shard_index * chunk_size
        # Se è l'ultimo shard, prende tutto il resto (per gestire i resti della divisione)
        end_idx = start_idx + chunk_size if shard_index < num_shards - 1 else len(all_keys)
        
        # 3. Selezioniamo solo i concetti assegnati a noi
        my_keys = all_keys[start_idx:end_idx]
        
        print(f"🖥️  Worker {shard_index}/{num_shards} - Assegnati {len(my_keys)} concetti su {len(all_keys)} totali.")
        stats = {"total": len(my_keys), "success": 0, "failed": 0}
        
        # Target context dal Config
        target_context = Config.get_background_template()
                
        for concept_id in my_keys:
            content = concept_dict[concept_id]
            print(f"\n🎨 Elaborazione concetto: {concept_id}")
            try:
                # 1. Recupero dell'immagine sorgente
                # Gestiamo sia il formato vecchio che nuovo
                images = content.get("image", content.get("selected_images", []))
                if not images:
                    print(f"   ⚠️ Nessuna immagine sorgente trovata per {concept_id}.")
                    continue
                source_image_path = images[0]
                
                # 2. Recupero degli attributi (Fingerprints)
                attributes = content.get("info", {})
                
                # 3. Textual Anchoring: Costruzione del prompt FLUX
                flux_prompt = build_flux_prompt(attributes, target_context)
                print(f"   📝 Prompt: {flux_prompt[:100]}...")
                
                # 4. Caricamento e standardizzazione immagine
                seed_img = Image.open(source_image_path).convert("RGB")
                # Fissiamo la dimensione per evitare che immagini troppo grosse causino OOM
                target_size = Config.Images.REFERENCE_IMAGE_SIZE
                seed_img = seed_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # Seme di generazione per la riproducibilità
                generator = torch.Generator(device=self.device).manual_seed(Config.Generate.SEED)
                
                # 5. La magia di FLUX Img2Img (Senza IP-Adapter, 4 step fissi)
                # Omettiamo 'strength' per far applicare a Diffusers il noise default (~0.8)
                output_img = self.pipe(
                    prompt=flux_prompt,
                    image=seed_img,
                    num_inference_steps=4,
                    generator=generator
                ).images[0]
                
                # 6. Salvataggio Output (Rispettiamo la nomenclatura di R2P-GEN originale)
                # In full_loop.py il file verificato si aspetta di chiamarsi: "{concept_id}_generated.png"
                out_path = os.path.join(self.output_dir, f"{concept_id}_generated.png")
                output_img.save(out_path)
                
                stats["success"] += 1
                print(f"   ✅ Generato e salvato: {os.path.basename(out_path)}")
                
            except Exception as e:
                print(f"   ❌ Errore durante la generazione di {concept_id}: {str(e)}")
                stats["failed"] += 1
                
        return stats

    def cleanup(self):
        """Libera rigorosamente la memoria."""
        print("\n🧹 Pulizia VRAM da FLUX...")
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        torch.cuda.empty_cache()