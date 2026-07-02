#!/usr/bin/env python3
"""
R2P-GEN Recovery Dashboard — v2

Mostra tre gruppi:
  ✅ PASSED     — ha superato il verify al primo colpo
  🔄 RECOVERED  — è entrato nel refine ed è stato salvato
  💀 GRAVEYARD  — è entrato nel refine ma non è stato recuperato

Per ogni concept mostra una timeline orizzontale:
  [Reference] → [Generata originale] → [Tentativo 1] → … → [Finale / Graveyard]

Con log completo del verify (fasi, VLM per attributo, CLIP breakdown, metodo di decisione).

Usage:
    python build_recovery_dashboard.py <run_dir>
    python build_recovery_dashboard.py /path/to/output/test_100

Legge (tutti dalla run_dir):
    prompts.json            — source_image (reference) + output_image (prima generata)
    recovery_results.json   — esiti refine per-concept + attempts_log
    rejected_concepts.json  — dettagli verify base (vlm_history, clip_details, method…)

Genera:
    <run_dir>/recovery_dashboard.html
    <run_dir>/dash_assets/   — symlink/copie immagini
"""

import json
import os
import sys
import shutil
import html as html_lib


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  [WARNING] File non trovato: {path}")
    return {}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def h(s) -> str:
    return html_lib.escape(str(s) if s is not None else "")


def fmt(v, d=3) -> str:
    if isinstance(v, float):
        return f"{v:.{d}f}"
    if v is None:
        return "N/A"
    return str(v)


def score_color(v) -> str:
    if v is None:
        return "#6e7681"
    if v >= 0.75:
        return "#3fb950"
    if v >= 0.45:
        return "#ffa657"
    return "#f85149"


def link_image(src: str, dst: str) -> bool:
    """Crea symlink (o copia fallback) da src a dst. Ritorna True se ok."""
    if not src or not os.path.exists(src):
        return False
    try:
        if os.path.lexists(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)
        return True
    except Exception:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"  [ERROR] link_image {src}: {e}")
            return False


def find_rejected_original(output_dir: str, concept_id: str) -> str:
    """
    Cerca il file <concept_id>_generated_rejected_attempt0.png nell'output dir.
    Questo esiste solo se il concept è entrato nel refine (flux_loop.py lo crea
    quando sovrascrive _generated.png con il recovered).
    """
    path = os.path.join(output_dir, f"{concept_id}_generated_rejected_attempt0.png")
    if os.path.exists(path):
        return path
    return ""


def find_latest_run(root_dir: str) -> str | None:
    output_dir = os.path.join(root_dir, "output")
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for d in os.listdir(output_dir):
        candidate = os.path.join(output_dir, d)
        rec_file = os.path.join(candidate, "recovery_results.json")
        if os.path.isfile(rec_file):
            candidates.append((os.path.getmtime(rec_file), candidate))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


# ─────────────────────────────────────────────────────────────
# HTML: singola cella della timeline
# ─────────────────────────────────────────────────────────────

def img_cell(label: str, img_path: str, assets_abs: str, assets_rel: str,
             safe_name: str, suffix: str, extra_css: str = "") -> str:
    """
    Rende una cella della timeline: etichetta + immagine.
    Crea il symlink in assets_abs, usa il path relativo assets_rel nel HTML.
    """
    ext = os.path.splitext(img_path)[1] if img_path else ".png"
    local_fn  = f"{safe_name}__{suffix}{ext}"
    local_abs = os.path.join(assets_abs, local_fn)
    local_rel = f"{assets_rel}/{local_fn}"
    ok = link_image(img_path, local_abs)

    img_tag = (
        f'<img src="{local_rel}" alt="{h(label)}" loading="lazy">'
        if ok else
        '<div class="img-placeholder">N/D</div>'
    )
    return f"""
    <div class="tl-cell {extra_css}">
        <div class="tl-label">{h(label)}</div>
        <div class="tl-img">{img_tag}</div>
    </div>"""


def arrow_cell() -> str:
    return '<div class="tl-arrow">→</div>'


# ─────────────────────────────────────────────────────────────
# HTML: blocco verify completo
# ─────────────────────────────────────────────────────────────

def build_verify_block(reject_data: dict) -> str:
    """
    Costruisce il blocco con tutti i dettagli del verify base dal rejected_concepts.json.
    Mostra: metodo di decisione, score, VLM history (tutte le fasi), CLIP breakdown.
    """
    if not reject_data:
        return ""

    details = reject_data.get("details", {})
    if not details:
        # Formato minimale (concept non aveva 'details')
        score   = reject_data.get("score", None)
        missing = reject_data.get("missing_details", [])
        tags    = "".join(f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in missing)
        return f"""
        <details class="section-toggle">
            <summary>🔎 Verify base</summary>
            <div class="toggle-body">
                <div class="kv-row"><span class="kv-key">Score</span><span class="kv-val" style="color:{score_color(score)}">{fmt(score)}</span></div>
                <div class="block-label" style="margin-top:8px">Attributi mancanti</div>
                <div style="margin-top:4px">{tags or '<em>—</em>'}</div>
            </div>
        </details>"""

    method   = details.get("method", "?")
    reason   = details.get("reason", "")
    score    = details.get("score", None)
    method_color = "#3fb950" if details.get("is_verified") else "#f85149"

    # ── VLM history ──────────────────────────────────────────
    vlm_history = details.get("vlm_history", [])

    # Raggruppa per fase
    phases = {}
    for entry in vlm_history:
        phase = entry.get("phase", "unknown")
        phases.setdefault(phase, []).append(entry)

    vlm_rows = ""
    for phase, entries in phases.items():
        phase_label = {
            "single_check": "Phase 1 — VLM Sweep (single image)",
            "pairwise":     "Phase 4 — Pairwise Comparison",
        }.get(phase, phase)

        vlm_rows += f'<tr><td colspan="4" class="phase-header">{h(phase_label)}</td></tr>'
        for e in entries:
            attr    = e.get("attribute", "?")
            yes_c   = e.get("yes_conf", e.get("score", 0.0))
            no_c    = e.get("no_conf", 0.0)
            resp    = e.get("response", "")[:80]
            row_col = score_color(yes_c)
            vlm_rows += f"""<tr>
                <td>{h(attr)}</td>
                <td style="color:{row_col};font-weight:600">{fmt(yes_c)}</td>
                <td style="color:{score_color(1-yes_c)}">{fmt(no_c)}</td>
                <td style="color:#6e7681;font-size:0.75rem">{h(resp)}</td>
            </tr>"""

    vlm_table = f"""
    <table class="data-table" style="margin-top:8px">
        <thead><tr><th>Attributo</th><th>Yes ↑</th><th>No</th><th>Risposta raw</th></tr></thead>
        <tbody>{vlm_rows}</tbody>
    </table>""" if vlm_rows else "<em style='color:#6e7681'>Nessun log VLM disponibile.</em>"

    # ── CLIP breakdown ────────────────────────────────────────
    clip_gen = details.get("clip_details", {}).get("gen", {})
    clip_ref = details.get("clip_details", {}).get("ref", {})
    clip_rows = ""
    for attr, gen_val in clip_gen.items():
        ref_val = clip_ref.get(attr, 0.0)
        delta   = gen_val - ref_val
        d_color = "#3fb950" if delta >= -0.03 else "#f85149"
        sign    = "+" if delta >= 0 else ""
        clip_rows += f"""<tr>
            <td>{h(attr)}</td>
            <td style="color:{score_color(gen_val)}">{fmt(gen_val)}</td>
            <td>{fmt(ref_val)}</td>
            <td style="color:{d_color};font-weight:600">{sign}{fmt(delta)}</td>
        </tr>"""

    clip_table = f"""
    <table class="data-table" style="margin-top:8px">
        <thead><tr><th>Attributo</th><th>Gen</th><th>Ref</th><th>Δ</th></tr></thead>
        <tbody>{clip_rows}</tbody>
    </table>""" if clip_rows else "<em style='color:#6e7681'>Nessun dato CLIP.</em>"

    # ── Attributi mancanti originali ──────────────────────────
    failed_attrs = details.get("failed_attributes", reject_data.get("missing_details", []))
    attr_tags = "".join(f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in failed_attrs) or "<em>—</em>"

    return f"""
    <details class="section-toggle">
        <summary>🔎 Verify base — dettaglio completo</summary>
        <div class="toggle-body">
            <div class="kv-grid">
                <div class="kv-row"><span class="kv-key">Metodo decisione</span><span class="kv-val" style="color:{method_color};font-weight:700">{h(method)}</span></div>
                <div class="kv-row"><span class="kv-key">Score finale</span><span class="kv-val" style="color:{score_color(score)}">{fmt(score)}</span></div>
                <div class="kv-row"><span class="kv-key">Reason</span><span class="kv-val" style="color:#8b949e">{h(reason)}</span></div>
            </div>

            <div class="block-label" style="margin-top:12px">Attributi falliti</div>
            <div style="margin-top:4px">{attr_tags}</div>

            <details class="section-toggle" style="margin-top:12px">
                <summary>📊 VLM log completo ({len(vlm_history)} check)</summary>
                <div class="toggle-body">{vlm_table}</div>
            </details>

            <details class="section-toggle" style="margin-top:8px">
                <summary>📐 CLIP breakdown per attributo</summary>
                <div class="toggle-body">{clip_table}</div>
            </details>
        </div>
    </details>"""


# ─────────────────────────────────────────────────────────────
# HTML: blocco attempt log del refine
# ─────────────────────────────────────────────────────────────

def build_attempt_details(attempt: dict) -> str:
    """Dettagli di un singolo tentativo (prompt, missing, metriche)."""
    prompt   = attempt.get("prompt_used", "")
    missing  = attempt.get("missing_attributes", [])
    clip     = attempt.get("clip_score", None)
    vlm      = attempt.get("vlm_avg", None)
    success  = attempt.get("success", False)

    attr_tags = "".join(
        f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in missing
    ) or '<span class="attr-tag attr-ok">✅ nessuno</span>'

    metrics = f"""
    <div class="kv-grid" style="margin-bottom:6px">
        <div class="kv-row"><span class="kv-key">CLIP score</span><span class="kv-val" style="color:{score_color(clip)}">{fmt(clip)}</span></div>
        <div class="kv-row"><span class="kv-key">VLM avg</span><span class="kv-val" style="color:{score_color(vlm)}">{fmt(vlm)}</span></div>
    </div>"""

    return f"""
    <div class="toggle-body">
        {metrics}
        <div class="block-label">Attributi ancora mancanti dopo questo tentativo</div>
        <div style="margin-top:4px;margin-bottom:8px">{attr_tags}</div>
        <div class="block-label">Prompt usato</div>
        <div class="prompt-text" style="margin-top:4px">{h(prompt)}</div>
    </div>"""


# ─────────────────────────────────────────────────────────────
# HTML: timeline completa per un concept
# ─────────────────────────────────────────────────────────────

def build_timeline(concept_id: str, safe_name: str, status: str,
                   first_gen_path: str, attempts_log: list,
                   final_path: str, assets_abs: str, assets_rel: str) -> str:
    """
    Costruisce la timeline orizzontale:
      [Generata orig] → [Attempt 1] → [Attempt 2] → … → [Finale / Graveyard]

    first_gen_path : output_image da prompts.json (la prima generata, prima del refine)
    attempts_log   : lista dei tentativi dal recovery_results.json
    final_path     : recovered_image_path (None se graveyard)

    Per i PASSED non c'è refine quindi la timeline è solo [Prima generata].
    """
    cells = []

    # Cella 0 — immagine generata originale (quella che ha fallito il verify, o la finale se passed)
    orig_label = "Generata originale"
    # Se il concept è entrato nel refine, esiste anche _generated_rejected_attempt0.png.
    # Altrimenti usiamo output_image da prompts.json.
    rejected0 = find_rejected_original(os.path.dirname(assets_abs.rstrip("/")), concept_id)
    # assets_abs è dentro run_dir, quindi torniamo su di un livello
    run_dir = os.path.dirname(assets_abs)
    rejected0 = find_rejected_original(run_dir, concept_id)

    orig_path = rejected0 if rejected0 else first_gen_path
    cells.append(img_cell(orig_label, orig_path, assets_abs, assets_rel, safe_name, "orig"))

    # Celle per ogni tentativo del refine
    for att in attempts_log:
        n        = att.get("attempt", "?")
        img_path = att.get("image_path", "")
        success  = att.get("success", False)
        label    = f"Attempt {n} {'✅' if success else '❌'}"
        extra    = "cell-success" if success else "cell-fail"
        cells.append(arrow_cell())
        cells.append(img_cell(label, img_path, assets_abs, assets_rel, safe_name, f"att{n}", extra))

    # Cella finale (solo se recovered e diversa dall'ultimo attempt)
    if status == "recovered" and final_path and attempts_log:
        last_att_path = attempts_log[-1].get("image_path", "")
        if os.path.abspath(final_path) != os.path.abspath(last_att_path or ""):
            cells.append(arrow_cell())
            cells.append(img_cell("✅ Finale accettata", final_path,
                                  assets_abs, assets_rel, safe_name, "final", "cell-success"))

    return f'<div class="timeline">{"".join(cells)}</div>'


# ─────────────────────────────────────────────────────────────
# HTML: card per ogni gruppo
# ─────────────────────────────────────────────────────────────

def build_passed_card(concept: str, safe_name: str, p_data: dict,
                      assets_abs: str, assets_rel: str) -> str:
    ref_path = p_data.get("source_image", "")
    gen_path = p_data.get("output_image", "")
    prompt   = p_data.get("flux_prompt", "")

    ref_cell = img_cell("📷 Reference", ref_path, assets_abs, assets_rel, safe_name, "ref")
    gen_cell = img_cell("✅ Generata", gen_path, assets_abs, assets_rel, safe_name, "gen", "cell-success")

    return f"""
    <div class="card passed" data-status="passed" data-concept="{h(safe_name)}">
        <div class="card-header">
            <code class="concept-name">{h(concept)}</code>
            <span class="badge badge-passed">✅ PASSED — primo verify</span>
        </div>
        <div class="timeline">
            {ref_cell}
            {arrow_cell()}
            {gen_cell}
        </div>
        <details class="section-toggle" style="margin-top:10px">
            <summary>📝 Prompt usato</summary>
            <div class="toggle-body">
                <div class="prompt-text">{h(prompt)}</div>
            </div>
        </details>
    </div>"""


def build_recovered_card(concept: str, safe_name: str,
                         p_data: dict, rec: dict, reject_data: dict,
                         assets_abs: str, assets_rel: str) -> str:
    concept_id      = safe_name
    ref_path        = p_data.get("source_image", "")
    first_gen_path  = p_data.get("output_image", "")
    final_path      = rec.get("recovered_image_path", "")
    attempts_log    = rec.get("attempts_log", [])
    orig_prompt     = rec.get("original_prompt", "")
    rewritten       = rec.get("last_rewritten_prompt", "")
    original_fail   = rec.get("original_fail_reason", [])
    n_attempts      = rec.get("attempts", len(attempts_log))

    fail_tags = "".join(f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in original_fail)

    ref_cell  = img_cell("📷 Reference", ref_path, assets_abs, assets_rel, safe_name, "ref")
    timeline  = build_timeline(concept_id, safe_name, "recovered",
                               first_gen_path, attempts_log, final_path,
                               assets_abs, assets_rel)
    verify_block   = build_verify_block(reject_data)

    # Attempt details accordion
    att_details = ""
    for att in attempts_log:
        n       = att.get("attempt", "?")
        success = att.get("success", False)
        label   = f"Attempt {n} — {'✅ SUCCESS' if success else '❌ fail'}"
        col     = "#3fb950" if success else "#f85149"
        att_details += f"""
        <details class="section-toggle">
            <summary style="color:{col}">{label}</summary>
            {build_attempt_details(att)}
        </details>"""

    return f"""
    <div class="card recovered" data-status="recovered" data-concept="{h(safe_name)}">
        <div class="card-header">
            <code class="concept-name">{h(concept)}</code>
            <span class="badge badge-recovered">🔄 RECOVERED in {n_attempts} attempt{"s" if n_attempts != 1 else ""}</span>
        </div>

        <div class="two-col">
            <div>
                <div class="tl-label" style="margin-bottom:6px">📷 Reference</div>
                {ref_cell.replace('class="tl-cell "', 'style="display:block"')}
            </div>
            <div style="flex:3">
                <div class="tl-label" style="margin-bottom:6px">Timeline generazione</div>
                {timeline}
            </div>
        </div>

        <div class="block-label" style="margin-top:12px">Attributi che mancavano</div>
        <div style="margin-top:4px;margin-bottom:10px">{fail_tags or '<em>—</em>'}</div>

        {verify_block}

        <details class="section-toggle" style="margin-top:8px">
            <summary>📝 Prompt: originale → rewritten</summary>
            <div class="toggle-body">
                <div class="block-label">Originale</div>
                <div class="prompt-text">{h(orig_prompt)}</div>
                <div class="block-label" style="margin-top:8px">Ultimo rewrite</div>
                <div class="prompt-text" style="color:#a5d6ff">{h(rewritten)}</div>
            </div>
        </details>

        <details class="section-toggle" style="margin-top:8px">
            <summary>🔁 Dettaglio attempt per attempt ({n_attempts})</summary>
            <div class="toggle-body">{att_details}</div>
        </details>
    </div>"""


def build_graveyard_card(concept: str, safe_name: str,
                         p_data: dict, rec: dict, reject_data: dict,
                         assets_abs: str, assets_rel: str) -> str:
    concept_id     = safe_name
    ref_path       = p_data.get("source_image", "")
    first_gen_path = p_data.get("output_image", "")
    attempts_log   = rec.get("attempts_log", [])
    orig_prompt    = rec.get("original_prompt", "")
    rewritten      = rec.get("last_rewritten_prompt", "")
    original_fail  = rec.get("original_fail_reason", [])
    last_missing   = rec.get("last_missing_details", [])
    final_score    = rec.get("final_score", None)
    reason         = rec.get("reason", "Max attempts reached")
    n_attempts     = rec.get("attempts", len(attempts_log))

    fail_tags = "".join(f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in original_fail)
    last_tags = "".join(f'<span class="attr-tag attr-fail">{h(a)}</span>' for a in last_missing) \
                or '<span class="attr-tag attr-ok">✅ nessuno</span>'

    ref_cell = img_cell("📷 Reference", ref_path, assets_abs, assets_rel, safe_name, "ref")
    timeline = build_timeline(concept_id, safe_name, "unrecoverable",
                              first_gen_path, attempts_log, None,
                              assets_abs, assets_rel)
    verify_block = build_verify_block(reject_data)

    att_details = ""
    for att in attempts_log:
        n       = att.get("attempt", "?")
        success = att.get("success", False)
        label   = f"Attempt {n} — {'✅ SUCCESS' if success else '❌ fail'}"
        col     = "#3fb950" if success else "#f85149"
        att_details += f"""
        <details class="section-toggle">
            <summary style="color:{col}">{label}</summary>
            {build_attempt_details(att)}
        </details>"""

    return f"""
    <div class="card graveyard" data-status="unrecoverable" data-concept="{h(safe_name)}">
        <div class="card-header">
            <code class="concept-name">{h(concept)}</code>
            <span class="badge badge-graveyard">💀 GRAVEYARD — {h(reason)}</span>
        </div>

        <div class="kv-grid" style="margin-bottom:10px">
            <div class="kv-row">
                <span class="kv-key">Final score</span>
                <span class="kv-val" style="color:{score_color(final_score)}">{fmt(final_score)}</span>
            </div>
            <div class="kv-row">
                <span class="kv-key">Attempts</span>
                <span class="kv-val" style="color:#79c0ff">{n_attempts}</span>
            </div>
        </div>

        <div class="two-col">
            <div>
                <div class="tl-label" style="margin-bottom:6px">📷 Reference</div>
                {ref_cell.replace('class="tl-cell "', 'style="display:block"')}
            </div>
            <div style="flex:3">
                <div class="tl-label" style="margin-bottom:6px">Timeline generazione</div>
                {timeline}
            </div>
        </div>

        <div style="margin-top:12px;display:flex;gap:24px;flex-wrap:wrap">
            <div>
                <div class="block-label">Mancava all'inizio</div>
                <div style="margin-top:4px">{fail_tags or '<em>—</em>'}</div>
            </div>
            <div>
                <div class="block-label">Mancava alla fine</div>
                <div style="margin-top:4px">{last_tags}</div>
            </div>
        </div>

        {verify_block}

        <details class="section-toggle" style="margin-top:8px">
            <summary>📝 Prompt: originale → ultimo rewrite</summary>
            <div class="toggle-body">
                <div class="block-label">Originale</div>
                <div class="prompt-text">{h(orig_prompt)}</div>
                <div class="block-label" style="margin-top:8px">Ultimo rewrite</div>
                <div class="prompt-text" style="color:#ffa657">{h(rewritten)}</div>
            </div>
        </details>

        <details class="section-toggle" style="margin-top:8px" open>
            <summary>🔁 Dettaglio attempt per attempt ({n_attempts})</summary>
            <div class="toggle-body">{att_details}</div>
        </details>
    </div>"""


# ─────────────────────────────────────────────────────────────
# CSS + JS
# ─────────────────────────────────────────────────────────────

CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: ui-monospace, "JetBrains Mono", "Fira Code", monospace;
    background: #0d1117;
    color: #c9d1d9;
    padding: 20px;
    font-size: 13px;
    line-height: 1.5;
}

/* ── Header ── */
.header {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 20px;
}
.header h1 { font-size: 1rem; font-weight: 700; color: #e6edf3; margin-bottom: 14px; }
.run-tag {
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.9em;
}

/* ── Stats ── */
.stats { display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap; }
.stat {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 10px 18px;
    text-align: center;
    min-width: 90px;
}
.stat-val { font-size: 1.7rem; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 0.68rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 3px; }

/* ── Filters ── */
.filters { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.filter-btn {
    padding: 4px 14px;
    border-radius: 20px;
    border: 1px solid #30363d;
    background: transparent;
    color: #8b949e;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.82rem;
    transition: all 0.12s;
}
.filter-btn:hover { border-color: #58a6ff; color: #58a6ff; }
.filter-btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; font-weight: 700; }
.search-box {
    margin-left: auto;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #1c2128;
    color: #c9d1d9;
    font-family: inherit;
    font-size: 0.82rem;
    width: 200px;
    outline: none;
}
.search-box:focus { border-color: #58a6ff; }

/* ── Section header ── */
.section-header {
    font-size: 0.85rem;
    font-weight: 700;
    padding: 8px 14px;
    border-radius: 8px;
    margin: 24px 0 12px;
    letter-spacing: 0.04em;
}
.hdr-passed      { background: #3fb95015; color: #3fb950; border: 1px solid #3fb95040; }
.hdr-recovered   { background: #79c0ff15; color: #79c0ff; border: 1px solid #79c0ff40; }
.hdr-graveyard   { background: #f8514915; color: #f85149; border: 1px solid #f8514940; }

/* ── Grid ── */
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(700px, 1fr));
    gap: 16px;
}

/* ── Card ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 18px;
    border-left: 3px solid #30363d;
}
.card:hover { border-color: #58a6ff44; }
.card.passed     { border-left-color: #3fb950; }
.card.recovered  { border-left-color: #79c0ff; }
.card.graveyard  { border-left-color: #f85149; }
.card.hidden     { display: none !important; }

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    flex-wrap: wrap;
    gap: 6px;
}
.concept-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #79c0ff;
    background: #1c2128;
    border: 1px solid #30363d;
    padding: 2px 8px;
    border-radius: 4px;
}
.badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-passed    { background: #3fb95018; color: #3fb950; border: 1px solid #3fb95055; }
.badge-recovered { background: #79c0ff18; color: #79c0ff; border: 1px solid #79c0ff55; }
.badge-graveyard { background: #f8514918; color: #f85149; border: 1px solid #f8514955; }

/* ── Two-col layout ── */
.two-col { display: flex; gap: 14px; align-items: flex-start; }

/* ── Timeline ── */
.timeline {
    display: flex;
    align-items: flex-start;
    gap: 0;
    overflow-x: auto;
    padding-bottom: 6px;
}
.tl-cell {
    display: flex;
    flex-direction: column;
    align-items: center;
    min-width: 160px;
    max-width: 200px;
    flex-shrink: 0;
}
.tl-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #6e7681;
    margin-bottom: 5px;
    text-align: center;
}
.tl-img img {
    width: 100%;
    max-height: 180px;
    object-fit: contain;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #0d1117;
    display: block;
}
.tl-img .img-placeholder {
    width: 140px;
    height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed #30363d;
    border-radius: 6px;
    color: #484f58;
    font-size: 0.8rem;
}
.cell-success .tl-label { color: #3fb950; }
.cell-success .tl-img img { border-color: #3fb95060; }
.cell-fail .tl-label    { color: #f85149; }
.cell-fail .tl-img img  { border-color: #f8514960; }

.tl-arrow {
    font-size: 1.4rem;
    color: #30363d;
    padding: 0 4px;
    align-self: center;
    flex-shrink: 0;
    margin-top: -16px; /* allinea visivamente al centro immagine */
}

/* ── KV rows ── */
.kv-grid { display: flex; flex-wrap: wrap; gap: 6px 24px; }
.kv-row  { display: flex; align-items: center; gap: 8px; font-size: 0.82rem; }
.kv-key  { color: #6e7681; min-width: 120px; }
.kv-val  { font-weight: 600; color: #c9d1d9; }

/* ── Toggle/accordion ── */
.section-toggle {
    margin-top: 10px;
    border-top: 1px solid #21262d;
    padding-top: 8px;
}
.section-toggle summary {
    cursor: pointer;
    color: #6e7681;
    font-size: 0.8rem;
    user-select: none;
    padding: 2px 0;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
}
.section-toggle summary:hover { color: #8b949e; }
.section-toggle summary::before { content: "▶"; font-size: 0.62rem; transition: transform 0.15s; }
.section-toggle[open] summary::before { transform: rotate(90deg); }

.toggle-body { margin-top: 10px; }

/* ── Prompt text ── */
.prompt-text {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 0.8rem;
    color: #a5d6ff;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Tags ── */
.attr-tag {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.74rem;
    margin: 1px 2px;
}
.attr-fail { background: #f8514915; color: #f85149; border: 1px solid #f8514940; }
.attr-ok   { background: #3fb95015; color: #3fb950; border: 1px solid #3fb95040; }

/* ── Block label ── */
.block-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #484f58;
    font-weight: 700;
}

/* ── Tables ── */
.data-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.data-table th {
    background: #1c2128;
    color: #6e7681;
    padding: 6px 10px;
    text-align: left;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #30363d;
}
.data-table td {
    padding: 5px 10px;
    border-bottom: 1px solid #21262d;
    color: #c9d1d9;
    vertical-align: top;
}
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover td { background: #1c212855; }
.phase-header {
    background: #1c2128;
    color: #6e7681;
    font-size: 0.72rem;
    font-style: italic;
    padding: 6px 10px;
    border-top: 1px solid #30363d;
}

.no-results {
    text-align: center;
    padding: 40px 20px;
    color: #484f58;
    grid-column: 1 / -1;
}
"""

JS = """
let currentFilter = 'all';
let currentSearch = '';

function applyFilter(status, btn) {
    currentFilter = status;
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    refresh();
}
function applySearch(val) {
    currentSearch = val.toLowerCase().trim();
    refresh();
}
function refresh() {
    document.querySelectorAll('.card').forEach(card => {
        const statusOk = currentFilter === 'all' || card.dataset.status === currentFilter;
        const searchOk = !currentSearch || card.dataset.concept.includes(currentSearch);
        card.classList.toggle('hidden', !(statusOk && searchOk));
    });
    // Mostra/nascondi section headers
    [['passed','passed'],['recovered','recovered'],['unrecoverable','graveyard']].forEach(([status, cls]) => {
        const hdr = document.getElementById('hdr-' + cls);
        const grid = document.getElementById('grid-' + cls);
        if (!hdr) return;
        const show = currentFilter === 'all' || currentFilter === status;
        hdr.style.display  = show ? '' : 'none';
        grid.style.display = show ? '' : 'none';
    });
}
"""


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def build_dashboard(run_dir: str):
    run_dir = os.path.abspath(run_dir)

    # Auto-detect se mancano i file obbligatori
    if not os.path.isfile(os.path.join(run_dir, "prompts.json")):
        print(f"  [INFO] prompts.json non trovato in {run_dir}, cerco ultima run...")
        latest = find_latest_run(run_dir)
        if latest:
            print(f"  [INFO] Trovata run: {latest}")
            run_dir = latest
        else:
            print(f"  [ERROR] Nessuna run valida trovata.")
            sys.exit(1)

    run_name   = os.path.basename(run_dir)
    assets_abs = os.path.join(run_dir, "dash_assets")
    assets_rel = "dash_assets"
    ensure_dir(assets_abs)

    print(f"\n=== R2P-GEN Recovery Dashboard v2 ===")
    print(f"Run: {run_name}  |  Dir: {run_dir}")

    prompts_data  = load_json(os.path.join(run_dir, "prompts.json"))
    recovery_data = load_json(os.path.join(run_dir, "recovery_results.json"))
    rejected_data = load_json(os.path.join(run_dir, "rejected_concepts.json"))

    print(f"  prompts.json       : {len(prompts_data)} entries")
    print(f"  recovery_results   : {len(recovery_data)} entries")
    print(f"  rejected_concepts  : {len(rejected_data)} entries")

    # ── Classifica ogni concept nei tre gruppi ──────────────────
    all_concepts = set(prompts_data.keys())
    recovery_concepts = set(recovery_data.keys())
    passed_concepts = all_concepts - recovery_concepts

    passed_cards     = []
    recovered_cards  = []
    graveyard_cards  = []

    # PASSED
    for concept in sorted(passed_concepts):
        safe = concept.replace("<", "").replace(">", "")
        p_data = prompts_data.get(concept, {})
        passed_cards.append(build_passed_card(concept, safe, p_data, assets_abs, assets_rel))

    # RECOVERED / GRAVEYARD
    for concept in sorted(recovery_concepts):
        safe      = concept.replace("<", "").replace(">", "")
        p_data    = prompts_data.get(concept, {})
        rec       = recovery_data[concept]
        reject_d  = rejected_data.get(concept, {})
        status    = rec.get("status", "unrecoverable")

        if status == "recovered":
            recovered_cards.append(
                build_recovered_card(concept, safe, p_data, rec, reject_d, assets_abs, assets_rel)
            )
        else:
            graveyard_cards.append(
                build_graveyard_card(concept, safe, p_data, rec, reject_d, assets_abs, assets_rel)
            )

    n_passed   = len(passed_cards)
    n_rec      = len(recovered_cards)
    n_grave    = len(graveyard_cards)
    n_total    = n_passed + n_rec + n_grave
    fail_rate  = round(n_grave * 100 / n_total) if n_total else 0

    print(f"  Passed: {n_passed}  |  Recovered: {n_rec}  |  Graveyard: {n_grave}")

    def section(hdr_id, grid_id, hdr_cls, hdr_label, cards):
        inner = "".join(cards) if cards else '<div class="no-results">Nessun concept in questo gruppo.</div>'
        return f"""
        <div class="section-header {hdr_cls}" id="{hdr_id}">{hdr_label}</div>
        <div class="grid" id="{grid_id}">{inner}</div>"""

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>R2P-GEN Dashboard — {h(run_name)}</title>
<style>{CSS}</style>
</head>
<body>

<div class="header">
    <h1>🔬 R2P-GEN Dashboard &nbsp;·&nbsp; <span class="run-tag">{h(run_name)}</span></h1>
    <div class="stats">
        <div class="stat"><div class="stat-val" style="color:#c9d1d9">{n_total}</div><div class="stat-lbl">Totale</div></div>
        <div class="stat"><div class="stat-val" style="color:#3fb950">{n_passed}</div><div class="stat-lbl">Passed</div></div>
        <div class="stat"><div class="stat-val" style="color:#79c0ff">{n_rec}</div><div class="stat-lbl">Recovered</div></div>
        <div class="stat"><div class="stat-val" style="color:#f85149">{n_grave}</div><div class="stat-lbl">Graveyard</div></div>
        <div class="stat"><div class="stat-val" style="color:#ffa657">{fail_rate}%</div><div class="stat-lbl">Fail rate</div></div>
    </div>
    <div class="filters">
        <button class="filter-btn active" onclick="applyFilter('all',this)">Tutti ({n_total})</button>
        <button class="filter-btn" onclick="applyFilter('passed',this)">✅ Passed ({n_passed})</button>
        <button class="filter-btn" onclick="applyFilter('recovered',this)">🔄 Recovered ({n_rec})</button>
        <button class="filter-btn" onclick="applyFilter('unrecoverable',this)">💀 Graveyard ({n_grave})</button>
        <input class="search-box" type="text" placeholder="🔍 cerca concept…" oninput="applySearch(this.value)">
    </div>
</div>

{section("hdr-graveyard", "grid-graveyard", "hdr-graveyard", f"💀 Graveyard &nbsp;·&nbsp; {n_grave} concept{'s' if n_grave!=1 else ''}", graveyard_cards)}
{section("hdr-recovered",  "grid-recovered",  "hdr-recovered",  f"🔄 Recovered &nbsp;·&nbsp; {n_rec} concept{'s' if n_rec!=1 else ''}",   recovered_cards)}
{section("hdr-passed",     "grid-passed",     "hdr-passed",     f"✅ Passed &nbsp;·&nbsp; {n_passed} concept{'s' if n_passed!=1 else ''}",  passed_cards)}

<script>{JS}</script>
</body>
</html>"""

    out_path = os.path.join(run_dir, "recovery_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ Dashboard pronta: {out_path}")
    print(f"\nPer visualizzare:")
    print(f"  cd '{run_dir}'")
    print(f"  python -m http.server 18082")
    print(f"  → http://localhost:18082/recovery_dashboard.html")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    build_dashboard(target)