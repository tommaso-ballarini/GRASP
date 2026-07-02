#!/usr/bin/env python3
"""
compare_ablations.py

Legge e confronta i risultati (final_judge_results.json) di diverse run di ablation.
Se 'final_score' non è presente, lo ricalcola come media delle metriche visive e testuali.
"""

import json
import sys
import os

# Definiamo le run e i percorsi
RUNS = {
    "A (Text Naive)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_A_text_naive/final_judge_results.json",
    "B (Text Fingerprints)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_B_text_fingerprints/final_judge_results.json",
    "C (Img+Txt Naive)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_C_image_text_naive/final_judge_results.json",
    "D (Img+Finger NoRef)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_D_image_fingerprints_norefine/final_judge_results.json",
    "E (Full Centroid)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_E_full_centroid/final_judge_results.json",
    "F (Full Textdriven)": "/leonardo/home/userexternal/tballari/R2P-GEN/output/ablation_full_F_full_textdriven/final_judge_results.json"
}

def percentile(sorted_vals: list, p: float) -> float:
    if not sorted_vals: return 0.0
    n = len(sorted_vals)
    idx = p / 100 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return sorted_vals[lo] + (idx - lo) * (sorted_vals[hi] - sorted_vals[lo])

def main():
    print(f"\n{'='*95}")
    print(f" R2P-GEN ABLATION STUDY COMPARISON")
    print(f"{'='*95}\n")

    results = {}
    
    # 1. Caricamento Dati
    for run_name, path in RUNS.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                results[run_name] = json.load(f)
        else:
            print(f"⚠️  [In attesa] {run_name} non trovato in {path}")

    if not results:
        print("❌ Nessun file json trovato. Esco.")
        sys.exit(1)

    print(f"\n✅ Caricate {len(results)} run completate.\n")

    stats = {}
    for run_name, data in results.items():
        scores = []
        metrics = {"clip_i": [], "clip_t": [], "dino_i": [], "tifa_score": []}
        
        for d in data.values():
            # Il dizionario metrics potrebbe essere vuoto o non esistere
            m = d.get("metrics", {})
            
            # Estrazione sicura delle metriche (ignorando i valori nulli/assenti)
            c_i = m.get("clip_i", 0.0)
            c_t = m.get("clip_t", 0.0)
            d_i = m.get("dino_i", 0.0)
            tifa = m.get("tifa_score", 0.0)
            
            # Popolamento liste per calcolo medie
            if c_i > 0: metrics["clip_i"].append(c_i)
            if c_t > 0: metrics["clip_t"].append(c_t)
            if d_i > 0: metrics["dino_i"].append(d_i)
            if tifa > 0: metrics["tifa_score"].append(tifa)

            # Ricalcolo FINAL SCORE se non esiste nel JSON principale
            # (Media di clip_i, dino_i e tifa_score. Adatta la formula se usavi pesi diversi)
            fs = d.get("final_score")
            if fs is None or fs == 0.0:
                # Calcoliamo la media solo sui valori presenti e maggiori di zero
                valid_vals = [v for v in [c_i, d_i, tifa] if v > 0]
                fs = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
                
            scores.append(fs)
            
        # Ordinamento obbligatorio per il calcolo dei percentili
        scores.sort()
        n = len(scores)
        if n == 0: continue
        
        mean = sum(scores) / n
        std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5
                    
        stats[run_name] = {
            "n": n,
            "mean": mean,
            "std": std,
            "p50": percentile(scores, 50),
            "p75": percentile(scores, 75),
            "scores": scores,
            "metrics_mean": {k: (sum(v)/len(v) if v else 0.0) for k, v in metrics.items()}
        }

    # Ordina per media decrescente
    sorted_runs = sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True)

    # ── TABELLA 1: Score Globale ───────────────────────────────────────────
    print(f" 📊 TABELLA 1: FINAL SCORE GLOBALE (Ordinata per Media Punteggio)")
    print(f" {'-'*85}")
    print(f" {'Run Name':<25} | {'N° Conc':<8} | {'Mean ± Std':<15} | {'P50 (Median)':<12} | {'P75':<10}")
    print(f" {'-'*85}")
    for run_name, s in sorted_runs:
        print(f" {run_name:<25} | {s['n']:<8} | {s['mean']:.3f} ± {s['std']:.3f} | {s['p50']:.3f}        | {s['p75']:.3f}")

    # ── TABELLA 2: Scomposizione Metriche ──────────────────────────────────
    print(f"\n\n 🔍 TABELLA 2: METRICHE SPECIFICHE (Zero significa non calcolato)")
    print(f" {'-'*75}")
    print(f" {'Run Name':<25} | {'CLIP-I':<10} | {'DINO-I':<10} | {'CLIP-T':<10} | {'TIFA':<10}")
    print(f" {'-'*75}")
    for run_name, s in sorted_runs:
        m = s["metrics_mean"]
        # Formattazione condizionale: se è 0, metto un "-" per chiarezza visiva
        c_i = f"{m['clip_i']:.3f}" if m['clip_i'] > 0 else "   -"
        d_i = f"{m['dino_i']:.3f}" if m['dino_i'] > 0 else "   -"
        c_t = f"{m['clip_t']:.3f}" if m['clip_t'] > 0 else "   -"
        t_s = f"{m['tifa_score']:.3f}" if m['tifa_score'] > 0 else "   -"
        
        print(f" {run_name:<25} | {c_i:<10} | {d_i:<10} | {c_t:<10} | {t_s:<10}")

    # ── TABELLA 3: Pass Rate a Soglie Fisse ────────────────────────────────
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    print(f"\n\n 🏆 TABELLA 3: SURVIVAL RATE (% di concetti salvati a soglie fisse)")
    print(f" {'-'*95}")
    header = " | ".join([f"S_{t:.2f}" for t in thresholds])
    print(f" {'Run Name':<25} | {header}")
    print(f" {'-'*95}")
    for run_name, s in sorted_runs:
        rates = []
        for t in thresholds:
            pass_rate = sum(1 for score in s["scores"] if score >= t) / s["n"]
            rates.append(f"{pass_rate:>6.1%}")
        row = " | ".join(rates)
        print(f" {run_name:<25} | {row}")
        
    print(f"\n{'='*95}\n")

if __name__ == "__main__":
    main()