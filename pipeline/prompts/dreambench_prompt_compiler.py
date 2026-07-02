"""
pipeline/prompts/dreambench_prompt_compiler.py

Prompt Compiler per la Fase 3 (DreamBench zero-shot).

Riusa le STESSE convenzioni già adottate in pipeline/prompts/flux_prompts.py
(_DETAIL_TEMPLATES, _EXCLUDED_KEYS, _NEGATIVE_TRIGGERS, fallback generico)
per minimizzare le differenze di stile tra i due moduli.

Differenza principale rispetto a build_flux_prompt():
- build_flux_prompt() produce un PARAGRAFO ("A high-quality photograph of...").
- compile_subject_phrase() produce una NOUN PHRASE pulita, pensata per essere
  iniettata al posto di {0} nei prompt ufficiali DreamBench
  (es. "a {0} in the jungle" -> "a maroon backpack with colorful patches
  on the front pocket, photographed in the jungle").

OPZIONE A (scelta concordata): nessun placeholder <concept_id> nella frase
finale. Il VLM lo usa internamente per l'estrazione ("<backpack> is..."),
ma qui viene sempre rimosso: FLUX deve ricevere solo linguaggio naturale,
e per CLIP-T (Fase 4) un prompt "naturale" è l'unico comparabile con la
letteratura.

NOTE SU build_dreambench_prompt():
- Prompt 0-19 (recontextualization / accessorization): la scena viene
  separata dalla noun phrase con ", photographed <scene>". Questo impedisce
  a FLUX di interpretare attributi di scena (es. "blue house", "Eiffel Tower")
  come proprietà dell'oggetto (attribute bleeding).
- Prompt 20-24 (property modification: "a red {0}", "a cube shaped {0}"...):
  nessuna separazione — il modificatore deve restare attaccato al soggetto
  perché è proprio ciò che il prompt chiede di cambiare.
  Questi prompt vengono passati as-is con build_dreambench_prompt() usando
  is_property_modification=True.
"""

import re

# ---------------------------------------------------------------------------
# Mapping chiave grezza -> frase naturale (OBJECT)
# Identico a flux_prompts.py, per coerenza di stile.
# ---------------------------------------------------------------------------

_DETAIL_TEMPLATES_OBJECT = {
    "shape":              "with a {value} shape",
    "pattern":             "featuring a {value} pattern",
    "brand/text":          "with {value} visible",
    "distinct features":  "showing {value}",
}

# ---------------------------------------------------------------------------
# Mapping chiave grezza -> frase naturale (LIVING)
# Stesso pattern, nuove chiavi per il secondo schema (animali).
# ---------------------------------------------------------------------------

_DETAIL_TEMPLATES_LIVING = {
    "coat_and_color":       "with {value} coat",
    "facial_features":      "with {value}",
    "distinctive_markings": "showing {value}",
    "accessories":          "wearing {value}",
}

# Chiavi già usate nel "soggetto principale" (non vanno ripetute nei dettagli)
_EXCLUDED_KEYS_OBJECT = {"category", "color", "material", "general", "sdxl_prompt", "_entity_type"}
_EXCLUDED_KEYS_LIVING = {"category", "species_and_breed", "general", "sdxl_prompt", "_entity_type"}

# Stessi trigger negativi di flux_prompts.py
_NEGATIVE_TRIGGERS = [
    "no visible", "none", "n/a", "not readable",
    "unknown", "no brand", "no text",
]

# Pattern per rimuovere qualsiasi placeholder <concept_id> residuo
# (Opzione A: non deve mai arrivare a FLUX)
_CONCEPT_TAG_RE = re.compile(r"<[^<>]+>\s*")


def _is_negative_value(value: str) -> bool:
    value_lower = value.strip().lower()
    return any(trigger in value_lower for trigger in _NEGATIVE_TRIGGERS)


def _format_detail(key: str, value: str, templates: dict) -> str:
    template = templates.get(key)
    if template:
        return template.format(value=value)
    # stesso fallback generico di flux_prompts.py
    return f"with {value}"


def _strip_concept_tag(text: str) -> str:
    """Rimuove ogni occorrenza di <concept_id> e normalizza gli spazi."""
    cleaned = _CONCEPT_TAG_RE.sub("", text)
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# Compilatori per categoria (OBJECT / LIVING)
# ---------------------------------------------------------------------------

def _compile_object_phrase(attributes: dict) -> str:
    category = _strip_concept_tag(attributes.get("category", "object")).lower()
    color = attributes.get("color", "")
    material = attributes.get("material", "")

    subject_parts = [p for p in (color, material) if p and not _is_negative_value(p)]
    main_subject = " ".join(subject_parts + [category]).strip()

    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS_OBJECT or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value, _DETAIL_TEMPLATES_OBJECT))

    phrase = main_subject
    if details:
        phrase = f"{phrase} {', '.join(details)}"

    return _strip_concept_tag(phrase)


def _compile_living_phrase(attributes: dict) -> str:
    # species_and_breed es. "Dog, Shiba Inu" -> usato come soggetto principale
    species = attributes.get("species_and_breed") or attributes.get("category", "animal")
    species = _strip_concept_tag(species)
    # normalizza "Dog, Shiba Inu" -> "Shiba Inu dog" (più naturale per T5/FLUX)
    if "," in species:
        parts = [p.strip() for p in species.split(",", 1)]
        if len(parts) == 2:
            species = f"{parts[1]} {parts[0].lower()}"

    details = []
    for key, value in attributes.items():
        if key in _EXCLUDED_KEYS_LIVING or not value:
            continue
        if isinstance(value, str) and _is_negative_value(value):
            continue
        details.append(_format_detail(key, value, _DETAIL_TEMPLATES_LIVING))

    phrase = species
    if details:
        phrase = f"{phrase} {', '.join(details)}"

    return _strip_concept_tag(phrase)


def compile_subject_phrase(fingerprints: dict) -> str:
    """
    Punto di ingresso principale: dato il dizionario 'info' di un concept
    nel database, ritorna una noun phrase pulita in linguaggio naturale,
    pronta per sostituire {0} nei prompt ufficiali DreamBench.

    Esempi di output:
      OBJECT -> "maroon backpack with colorful patches on the front pocket"
      LIVING -> "Shiba Inu dog with short tan coat, white patch on the left paw"
    """
    entity_type = fingerprints.get("_entity_type", "OBJECT")
    if entity_type == "LIVING":
        phrase = _compile_living_phrase(fingerprints)
    else:
        phrase = _compile_object_phrase(fingerprints)

    # pulizia finale: niente doppi spazi, niente virgole pendenti
    phrase = " ".join(phrase.split())
    phrase = phrase.strip(" ,")
    return phrase


def build_dreambench_prompt(template: str, subject_phrase: str, is_property_modification: bool = False) -> str:
    """
    Inietta la subject_phrase nel template DreamBench (che usa {0}).

    Prompt 0-19 (recontextualization / accessorization):
        Separa la noun phrase dalla scena con ", photographed <scene>".
        Questo evita che FLUX interpreti attributi ambientali (es. "blue house",
        "Eiffel Tower") come proprietà dell'oggetto (attribute bleeding).

        Esempi:
          "a {0} in the jungle"
            -> "a <phrase>, photographed in the jungle"
          "a {0} with the Eiffel Tower in the background"
            -> "a <phrase>, photographed with the Eiffel Tower in the background"
          "a {0} on top of pink fabric"
            -> "a <phrase>, photographed on top of pink fabric"

    Prompt 20-24 (property modification: "a red {0}", "a cube shaped {0}"...):
        Nessuna separazione — il modificatore deve restare attaccato al soggetto
        perché è esattamente ciò che il prompt chiede di alterare.
        Passa is_property_modification=True per usare questo path.

        Esempio:
          "a red {0}" -> "a red <phrase>"
    """
    if is_property_modification:
        # Inserimento diretto: il modificatore fa parte del soggetto
        return template.format(subject_phrase)

    # Split prefix / scene sul placeholder {0}
    parts = template.split("{0}", 1)
    prefix = parts[0].strip()          # tipicamente "a"
    scene  = parts[1].strip() if len(parts) > 1 else ""

    if not scene:
        return f"{prefix} {subject_phrase}"

    return f"{prefix} {subject_phrase}, photographed {scene}"


# ==========================================
# ESEMPIO DI UTILIZZO (stesso stile del modulo originale)
# ==========================================
if __name__ == "__main__":
    object_fp = {
        "general": "<backpack> is a maroon-colored backpack with a front zipper pocket and several patches on its lower front.",
        "category": "backpack",
        "shape": "rectangular with rounded top corners",
        "material": "fabric",
        "color": "maroon",
        "pattern": "none",
        "brand/text": "none",
        "distinct features": "has multiple colorful patches on the front pocket",
        "_entity_type": "OBJECT",
    }

    living_fp = {
        "general": "<dog6> is a Shiba Inu with a thick tan and white coat.",
        "category": "dog",
        "species_and_breed": "Dog, Shiba Inu",
        "coat_and_color": "thick tan and white",
        "facial_features": "dark almond-shaped eyes, triangular upright ears",
        "distinctive_markings": "white patch on the left front paw",
        "accessories": "none",
        "_entity_type": "LIVING",
    }

    obj_phrase = compile_subject_phrase(object_fp)
    live_phrase = compile_subject_phrase(living_fp)

    print("OBJECT phrase:", obj_phrase)
    print("LIVING phrase:", live_phrase)
    print()

    # Recontextualization (prompt 0-19) — usa separazione fotografica
    recontex_templates = [
        "a {0} in the jungle",
        "a {0} on top of pink fabric",
        "a {0} with a blue house in the background",
        "a {0} with the Eiffel Tower in the background",
    ]
    print("--- Recontextualization ---")
    for t in recontex_templates:
        print(f"  {build_dreambench_prompt(t, obj_phrase)}")

    print()

    # Property modification (prompt 20-24) — inserimento diretto
    property_templates = [
        "a red {0}",
        "a purple {0}",
        "a cube shaped {0}",
    ]
    print("--- Property modification ---")
    for t in property_templates:
        print(f"  {build_dreambench_prompt(t, obj_phrase, is_property_modification=True)}")