"""
Costruzione del prompt FLUX (Textual Anchoring) a partire dagli attributi
estratti dal VLM.

FIX 6: le chiavi "grezze" del database (es. "brand/text", "distinct features")
non vengono più scritte letteralmente nel prompt come "brand/text like ...".
Vengono invece tradotte in frasi naturali tramite _DETAIL_TEMPLATES, con un
fallback generico per chiavi non previste.
"""

# ---------------------------------------------------------------------------
# Mapping chiave grezza -> frase naturale
# ---------------------------------------------------------------------------
# Ogni template riceve {value} = il contenuto del campo fingerprint.

_DETAIL_TEMPLATES = {
    "shape":             "with a {value} shape",
    "pattern":           "featuring a {value} pattern",
    "brand/text":        "with {value} visible",
    "distinct features": "showing {value}",
    "soft tags":         "also known as {value}",
    "state":             "{value}",
}

# Chiavi che NON vanno mai inserite nel loop dei dettagli extra:
# - category/color/material: già usati nella frase principale (subject)
# - general: riassunto discorsivo che duplica gli altri campi
# - sdxl_prompt: output di un'altra pipeline (SDXL), non un attributo
_EXCLUDED_KEYS = {"category", "color", "material", "general", "sdxl_prompt"}

# Valori "negativi" del VLM (campo non applicabile/non rilevato).
# Questi valori NON vanno inseriti nel prompt FLUX, altrimenti
# si ottengono frasi insensate come "featuring a no visible pattern pattern".
_NEGATIVE_TRIGGERS = [
    "no visible", "none", "n/a", "not readable",
    "unknown", "no brand", "no text",
]


def _is_negative_value(value: str) -> bool:
    """True se il valore indica un campo assente/non rilevato dal VLM."""
    value_lower = value.strip().lower()
    return any(trigger in value_lower for trigger in _NEGATIVE_TRIGGERS)


def _format_detail(key: str, value: str) -> str:
    """
    Traduce una coppia (chiave, valore) del fingerprint in una frase naturale.

    Usa _DETAIL_TEMPLATES se la chiave è nota, altrimenti applica un
    fallback generico che mantiene il valore ma scarta il nome-chiave grezzo.
    """
    template = _DETAIL_TEMPLATES.get(key)
    if template:
        return template.format(value=value)
    # Fallback generico per chiavi non previste: nessuna info persa,
    # ma niente "underscored_key like ..." nel prompt finale.
    return f"with {value}"


def build_flux_prompt(attributes: dict, target_context: str) -> str:
    """
    Costruisce un prompt in linguaggio naturale ottimizzato per il T5 encoder di FLUX,
    unendo un dizionario di attributi fisici (Textual Anchoring) con un contesto target.

    Args:
        attributes (dict): Dizionario degli attributi estratti dal VLM
                           (es. {"category": "sneakers", "color": "red", "material": "leather"}).
        target_context (str): La scena o il setup fotografico di destinazione.
                              (es. "placed on a wooden table, studio lighting").

    Returns:
        str: Un paragrafo fluido e descrittivo pronto per FLUX.
    """
    # 1. Estrazione sicura della categoria principale (default "object" se mancante)
    category = attributes.get("category", "object").lower()

    # 2. Costruzione discorsiva degli attributi principali (Colore e Materiale)
    # T5 preferisce frasi come "A red leather sneakers" piuttosto che "sneakers, red, leather"
    color = attributes.get("color", "")
    material = attributes.get("material", "")

    subject_parts = []
    if color:
        subject_parts.append(color)
    if material:
        subject_parts.append(material)

    # Uniamo colore e materiale prima della categoria
    subject_modifiers = " ".join(subject_parts)
    main_subject = f"{subject_modifiers} {category}".strip()

    # Iniziamo la frase in modo naturale
    prompt_sentences = [f"A high-quality photograph of a {main_subject}."]

    # 3. Aggiunta dei dettagli specifici (es. loghi, forme, pattern)
    # FIX 6: ogni chiave viene tradotta in una frase naturale tramite
    # _format_detail(), invece di scrivere "key like value" con la chiave
    # grezza del database.
    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value))

    if details:
        # Es: "It features Nike logo visible, and showing a scratch near the heel."
        prompt_sentences.append(f"It features {', and '.join(details)}.")

    # 4. Integrazione del target_context in modo fluido
    if target_context:
        # Assicuriamoci che il contesto si leghi bene all'oggetto
        clean_context = target_context.strip()
        if not clean_context.lower().startswith(("placed", "resting", "hanging", "in", "on", "with")):
            clean_context = f"The {category} is {clean_context}"
        else:
            clean_context = f"The {category} is {clean_context}"

        prompt_sentences.append(clean_context + ".")

    # Uniamo tutto in un singolo paragrafo coeso
    final_prompt = " ".join(prompt_sentences)

    # Pulizia di eventuali doppi spazi o punteggiatura ridondante
    final_prompt = " ".join(final_prompt.split())
    final_prompt = final_prompt.replace("..", ".")

    return final_prompt


# ==========================================
# ESEMPIO DI UTILIZZO
# ==========================================
if __name__ == "__main__":
    extracted_attributes = {
        "category": "sneakers",
        "color": "red with white stripes",
        "material": "leather",
        "shape": "low-top with rounded toe",
        "pattern": "no visible pattern",
        "brand/text": "a prominent Nike logo on the side",
        "distinct features": "a small scratch near the heel",
        "general": "<abc> is a red leather sneaker with white stripes.",
        "sdxl_prompt": "(red leather sneakers:1.3), Nike logo, ...",
    }
    context = "placed on a rustic wooden table, soft shadows, professional studio lighting"

    prompt = build_flux_prompt(extracted_attributes, context)
    print(prompt)
    # OUTPUT REALE:
    # A high-quality photograph of a red with white stripes leather sneakers.
    # It features with a low-top with rounded toe shape, and with a prominent
    # Nike logo on the side visible, and showing a small scratch near the heel.
    # The sneakers is placed on a rustic wooden table, soft shadows, professional
    # studio lighting.
    # Nota: "pattern": "no visible pattern" è stato scartato dal filtro
    # _is_negative_value (FIX 6).