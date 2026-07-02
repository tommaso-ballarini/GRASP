"""
pipeline/prompts/dreambench_prompts.py

Prompt ufficiali del benchmark DreamBooth/DreamBench (Ruiz et al., 2022).
25 prompt per gli OBJECT (20 recontextualization + 5 property modification)
25 prompt per i LIVING subjects (10 recontextualization + 10 accessorization + 5 property modification)

Ogni prompt usa il placeholder {0} per la classe/soggetto (es. "backpack", "dog").
Nel nostro pipeline {0} verrà sostituito dalla frase fluida compilata dal
Prompt Compiler (Fase 3), NON dal solo nome della classe.
"""

# ---------------------------------------------------------------------------
# OBJECT — 21 dei 30 soggetti DreamBench
# ---------------------------------------------------------------------------

OBJECT_PROMPTS = [
    # --- Recontextualization (20) ---
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    "a {0} with a wheat field in the background",
    "a {0} with a tree and autumn leaves in the background",
    "a {0} with the Eiffel Tower in the background",
    "a {0} floating on top of water",
    "a {0} floating in an ocean of milk",
    "a {0} on top of green grass with sunflowers around it",
    "a {0} on top of a mirror",
    "a {0} on top of the sidewalk in a crowded street",
    "a {0} on top of a dirt road",
    "a {0} on top of a white rug",
    # --- Property / color modification (5) ---
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]

# ---------------------------------------------------------------------------
# LIVING — 9 dei 30 soggetti DreamBench (cani/gatti)
# ---------------------------------------------------------------------------

LIVING_PROMPTS = [
    # --- Recontextualization (10) ---
    "a {0} in the jungle",
    "a {0} in the snow",
    "a {0} on the beach",
    "a {0} on a cobblestone street",
    "a {0} on top of pink fabric",
    "a {0} on top of a wooden floor",
    "a {0} with a city in the background",
    "a {0} with a mountain in the background",
    "a {0} with a blue house in the background",
    "a {0} on top of a purple rug in a forest",
    # --- Accessorization (10) ---
    "a {0} wearing a red hat",
    "a {0} wearing a santa hat",
    "a {0} wearing a rainbow scarf",
    "a {0} wearing a black top hat and a monocle",
    "a {0} in a chef outfit",
    "a {0} in a firefighter outfit",
    "a {0} in a police outfit",
    "a {0} wearing pink glasses",
    "a {0} wearing a yellow shirt",
    "a {0} in a purple wizard outfit",
    # --- Property modification (5) ---
    "a red {0}",
    "a purple {0}",
    "a shiny {0}",
    "a wet {0}",
    "a cube shaped {0}",
]


# ---------------------------------------------------------------------------
# Property modification prompts — ultimi 5 indici (20-24) in ENTRAMBE le liste.
# Questi prompt chiedono esplicitamente di cambiare colore/materiale/forma del
# soggetto ("a red {0}", "a cube shaped {0}", ...): sono in contraddizione
# logica con il verify per attributi (che controlla la fedeltà di colore/
# materiale/forma rispetto al reference). Per questo vanno esclusi dal verify
# e dal refine, e passano direttamente alla Fase 4 (metriche ufficiali).
#
# Nota: a differenza del single-pass "zero-shot puro", qui scegliamo
# DELIBERATAMENTE di applicare verify/refine sugli altri 20 prompt, perché il
# confronto è con DreamBooth (fine-tuned per soggetto) e non con un metodo
# training-free valutato solo zero-shot: anche il recovery loop fa parte del
# nostro metodo e va mostrato dove ha senso applicarlo.
# ---------------------------------------------------------------------------

PROPERTY_MODIFICATION_INDICES = set(range(20, 25))


def is_property_modification_prompt(prompt_idx: int) -> bool:
    """True se il prompt a questo indice è uno dei 5 di property modification
    (color/material/shape) e quindi va escluso dal verify/refine interno."""
    return prompt_idx in PROPERTY_MODIFICATION_INDICES


def get_prompts_for_entity_type(entity_type: str) -> list[str]:
    """
    Ritorna la lista di 25 prompt corretta in base al tipo di entità
    classificato in fase di estrazione fingerprint ('_entity_type' nel JSON).
    """
    if entity_type == "LIVING":
        return LIVING_PROMPTS
    return OBJECT_PROMPTS


if __name__ == "__main__":
    print(f"OBJECT_PROMPTS: {len(OBJECT_PROMPTS)} prompt")
    print(f"LIVING_PROMPTS: {len(LIVING_PROMPTS)} prompt")
    assert len(OBJECT_PROMPTS) == 25
    assert len(LIVING_PROMPTS) == 25
    print("OK: 25/25 per entrambe le categorie.")