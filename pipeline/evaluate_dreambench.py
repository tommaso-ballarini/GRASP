"""
pipeline/evaluate_dreambench.py

Fase 4 — Metriche ufficiali DreamBench per la tabella del paper.

Calcola DINO-I, CLIP-I, CLIP-T con i checkpoint allineati alla letteratura:
    - CLIP-I / CLIP-T : openai/clip-vit-base-patch32  (ViT-B/32)
    - DINO-I          : facebook/dino-vits16           (ViT-S/16, DINOv1)

Protocollo esattamente identico a DreamBooth (Ruiz et al., 2023):
    - DINO-I / CLIP-I : media pairwise tra ogni immagine generata e
                        TUTTE le immagini reali del soggetto nel database.
    - CLIP-T          : similarity tra immagine generata e template
                        DreamBench col nome-classe (NON la subject_phrase
                        arricchita inviata a FLUX — quel testo non è quello
                        che i paper comparabili usano per CLIP-T).

Produce due righe per la tabella:
    - zero_shot : immagini generate da generate_dreambench.py, PRIMA
                  del verify/refine (run principale, confrontabile con
                  DreamBooth e altri metodi zero-shot).
    - full      : immagini finali dopo il recovery loop di refine_dreambench.py
                  (usato come ablation row; disclosure metodologica nel paper).

I 5 prompt di property modification (indici 20-24) sono ESCLUSI
dal computo di DINO-I e CLIP-I (same as DreamBooth protocol, che
non valuta la fedeltà del soggetto quando il prompt chiede
esplicitamente di cambiarla), e INCLUSI solo in CLIP-T.

Output:
    {output_dir}/metrics_dreambench.json   — risultati per immagine
    {output_dir}/metrics_summary.json      — medie per soggetto e globali
    {output_dir}/metrics_table.txt         — tabella pronta per il paper
"""

import os
import sys
import json
import glob
import argparse
from tqdm import tqdm
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.metrics_dreambench import ClipDreamBench, DinoDreamBench
from pipeline.prompts.dreambench_prompts import (
    get_prompts_for_entity_type,
    is_property_modification_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize_folder_name(concept_id: str) -> str:
    return concept_id.strip("<>")


def _get_all_real_images(content: dict) -> list:
    """Ritorna TUTTE le immagini reali del soggetto (per pairwise).
    Usa content["image"] che il database popola già con l'intera cartella."""
    images = content.get("image", [])
    if isinstance(images, list) and images:
        return [p for p in images if os.path.exists(p)]
    # fallback su representative_image se per qualche motivo "image" è vuoto
    rep = content.get("representative_image")
    if rep and os.path.exists(rep):
        return [rep]
    return []


def _build_clipt_prompt(template: str, class_name: str) -> str:
    """Inietta il nome-classe (es. 'backpack_dog') nel template DreamBench.
    Non usiamo la subject_phrase arricchita: CLIP-T deve misurare
    l'allineamento col testo del benchmark, non col nostro prompt FLUX."""
    return template.format(class_name)


def _find_generated_images(output_dir: str, concept_name: str, prompt_idx: int) -> list:
    """Trova tutte le immagini generate per un dato (concept, prompt_idx).
    Esclude i residui *_rejected_attemptN.png lasciati da refine_dreambench.py
    (rename dell'originale prima di sovrascriverla con l'immagine recovered)."""
    prompt_dir = os.path.join(output_dir, concept_name, f"{prompt_idx:02d}")
    if not os.path.isdir(prompt_dir):
        return []
    all_pngs = glob.glob(os.path.join(prompt_dir, "*.png"))
    clean = [p for p in all_pngs if "_rejected_attempt" not in os.path.basename(p)]
    return sorted(clean)

# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    database_path: str,
    output_dir: str,
    results_dir: str,
    device: str = "cuda",
) -> dict:
    """Valuta tutte le immagini generate e ritorna il dizionario risultati."""

    print(f"\n{'='*70}")
    print("📊 FASE 4 — DREAMBENCH EVALUATION (DINO-I / CLIP-I / CLIP-T)")
    print(f"{'='*70}")
    print(f"   Database  : {database_path}")
    print(f"   Immagini  : {output_dir}")
    print(f"   Output    : {results_dir}")
    print()

    with open(database_path, "r", encoding="utf-8") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    # Carica i modelli una volta sola
    print("   Caricamento modelli metriche (checkpoint ufficiali DreamBench)...")
    clip_calc = ClipDreamBench(device=device)
    dino_calc = DinoDreamBench(device=device)
    print("   ✅ Modelli pronti.\n")

    # risultati per immagine
    per_image_results = {}

    for concept_id, content in tqdm(concept_dict.items(), desc="Concepts"):
        class_name   = content.get("name", _sanitize_folder_name(concept_id))
        entity_type  = content.get("info", {}).get("_entity_type", "OBJECT")
        real_images  = _get_all_real_images(content)
        prompts      = get_prompts_for_entity_type(entity_type)
        concept_name = _sanitize_folder_name(concept_id)  # nome cartella output

        if not real_images:
            print(f"   ⚠️  {concept_id}: nessuna immagine reale trovata → skip.")
            continue

        for prompt_idx, template in enumerate(prompts):
            is_prop_mod = is_property_modification_prompt(prompt_idx)
            clipt_prompt = _build_clipt_prompt(template, class_name)

            gen_images = _find_generated_images(output_dir, concept_name, prompt_idx)
            if not gen_images:
                continue

            for gen_path in gen_images:
                img_idx = os.path.splitext(os.path.basename(gen_path))[0]
                key = f"{concept_id}/{prompt_idx:02d}/{img_idx}"

                result = {
                    "concept_id":   concept_id,
                    "class_name":   class_name,
                    "entity_type":  entity_type,
                    "prompt_idx":   prompt_idx,
                    "img_idx":      img_idx,
                    "is_prop_mod":  is_prop_mod,
                    "clipt_prompt": clipt_prompt,
                    "gen_path":     gen_path,
                    "dino_i":       None,
                    "clip_i":       None,
                    "clip_t":       None,
                }

                # CLIP-T: calcolato sempre (inclusi i 5 prompt prop-mod)
                try:
                    result["clip_t"] = clip_calc.clip_t(gen_path, clipt_prompt)
                except Exception as e:
                    print(f"\n   ❌ CLIP-T errore su {key}: {e}")

                # DINO-I e CLIP-I: solo per i 20 prompt non-prop-mod
                if not is_prop_mod:
                    try:
                        result["clip_i"] = clip_calc.clip_i(gen_path, real_images)
                    except Exception as e:
                        print(f"\n   ❌ CLIP-I errore su {key}: {e}")
                    try:
                        result["dino_i"] = dino_calc.dino_i(gen_path, real_images)
                    except Exception as e:
                        print(f"\n   ❌ DINO-I errore su {key}: {e}")

                per_image_results[key] = result

    clip_calc.cleanup()
    dino_calc.cleanup()

    return per_image_results


# ---------------------------------------------------------------------------
# Aggregazione: media per subject e globale
# ---------------------------------------------------------------------------

def aggregate(per_image_results: dict) -> dict:
    """Calcola medie per soggetto e medie globali (overall).

    Struttura output:
        summary["per_concept"][concept_id] = {dino_i, clip_i, clip_t}
        summary["overall"]                 = {dino_i, clip_i, clip_t}
        summary["overall_living"]          = {dino_i, clip_i, clip_t}
        summary["overall_object"]          = {dino_i, clip_i, clip_t}
    """
    # accumulatori per concept
    concept_buckets = defaultdict(lambda: {
        "dino_i": [], "clip_i": [], "clip_t": [],
        "entity_type": None,
    })

    for key, r in per_image_results.items():
        cid = r["concept_id"]
        concept_buckets[cid]["entity_type"] = r["entity_type"]

        if r["clip_t"] is not None:
            concept_buckets[cid]["clip_t"].append(r["clip_t"])
        if not r["is_prop_mod"]:
            if r["dino_i"] is not None:
                concept_buckets[cid]["dino_i"].append(r["dino_i"])
            if r["clip_i"] is not None:
                concept_buckets[cid]["clip_i"].append(r["clip_i"])

    def _safe_mean(lst):
        return sum(lst) / len(lst) if lst else None

    per_concept = {}
    for cid, b in concept_buckets.items():
        per_concept[cid] = {
            "entity_type": b["entity_type"],
            "dino_i":      _safe_mean(b["dino_i"]),
            "clip_i":      _safe_mean(b["clip_i"]),
            "clip_t":      _safe_mean(b["clip_t"]),
            "n_dino":      len(b["dino_i"]),
            "n_clip_i":    len(b["clip_i"]),
            "n_clip_t":    len(b["clip_t"]),
        }

    # Medie globali (media delle medie per soggetto — protocollo DreamBooth)
    def _overall(filter_fn=None):
        dino_vals, clip_i_vals, clip_t_vals = [], [], []
        for cid, stats in per_concept.items():
            if filter_fn and not filter_fn(stats):
                continue
            if stats["dino_i"] is not None:
                dino_vals.append(stats["dino_i"])
            if stats["clip_i"] is not None:
                clip_i_vals.append(stats["clip_i"])
            if stats["clip_t"] is not None:
                clip_t_vals.append(stats["clip_t"])
        return {
            "dino_i": _safe_mean(dino_vals),
            "clip_i": _safe_mean(clip_i_vals),
            "clip_t": _safe_mean(clip_t_vals),
            "n_concepts": len(dino_vals),
        }

    return {
        "per_concept":     per_concept,
        "overall":         _overall(),
        "overall_living":  _overall(lambda s: s["entity_type"] == "LIVING"),
        "overall_object":  _overall(lambda s: s["entity_type"] == "OBJECT"),
    }


# ---------------------------------------------------------------------------
# Stampa tabella paper-ready
# ---------------------------------------------------------------------------

def print_table(summary: dict, run_label: str = "R2P-GEN") -> str:
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  DREAMBENCH RESULTS — {run_label}")
    lines.append(f"{'='*65}")
    lines.append(f"  {'':25s}  {'DINO-I':>8}  {'CLIP-I':>8}  {'CLIP-T':>8}")
    lines.append(f"  {'-'*55}")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A "

    ov = summary["overall"]
    lines.append(
        f"  {'Overall (all 30 subjects)':25s}  {fmt(ov['dino_i']):>8}"
        f"  {fmt(ov['clip_i']):>8}  {fmt(ov['clip_t']):>8}"
        f"  (n={ov['n_concepts']})"
    )

    ov_o = summary["overall_object"]
    lines.append(
        f"  {'  Objects (21)':25s}  {fmt(ov_o['dino_i']):>8}"
        f"  {fmt(ov_o['clip_i']):>8}  {fmt(ov_o['clip_t']):>8}"
    )

    ov_l = summary["overall_living"]
    lines.append(
        f"  {'  Living (9)':25s}  {fmt(ov_l['dino_i']):>8}"
        f"  {fmt(ov_l['clip_i']):>8}  {fmt(ov_l['clip_t']):>8}"
    )

    lines.append(f"\n  {'Per-subject breakdown':}")
    lines.append(f"  {'-'*55}")
    for cid, stats in sorted(summary["per_concept"].items()):
        tag = "L" if stats["entity_type"] == "LIVING" else "O"
        name = cid.strip("<>")[:22]
        lines.append(
            f"  [{tag}] {name:22s}  {fmt(stats['dino_i']):>8}"
            f"  {fmt(stats['clip_i']):>8}  {fmt(stats['clip_t']):>8}"
        )

    lines.append(f"{'='*65}\n")
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(
    database_path: str,
    output_dir: str,
    results_dir: str,
    run_label: str = "R2P-GEN",
    device: str = "cuda",
):
    os.makedirs(results_dir, exist_ok=True)

    # 1. Calcolo metriche per immagine
    per_image = evaluate(database_path, output_dir, results_dir, device=device)

    # 2. Salva risultati per immagine (utile per debug e analisi post-hoc)
    per_image_path = os.path.join(results_dir, "metrics_dreambench.json")
    with open(per_image_path, "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Risultati per immagine → {per_image_path}")

    # 3. Aggrega
    summary = aggregate(per_image)

    # 4. Salva summary
    summary_path = os.path.join(results_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 Summary → {summary_path}")

    # 5. Stampa e salva tabella
    table_str = print_table(summary, run_label=run_label)
    table_path = os.path.join(results_dir, "metrics_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"📄 Tabella → {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamBench Fase 4 — Evaluation")
    parser.add_argument("--database",     required=True,  help="Path al database_db.json")
    parser.add_argument("--output",       required=True,  help="Cartella con le immagini generate (generate_dreambench.py output)")
    parser.add_argument("--results-dir",  required=True,  help="Cartella dove salvare i JSON di metriche")
    parser.add_argument("--label",        default="R2P-GEN", help="Etichetta per la tabella (es. 'R2P-GEN (zero-shot)' o 'R2P-GEN (full)')")
    parser.add_argument("--device",       default="cuda")
    args = parser.parse_args()

    run_evaluation(
        database_path=args.database,
        output_dir=args.output,
        results_dir=args.results_dir,
        run_label=args.label,
        device=args.device,
    )