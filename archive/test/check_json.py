import json
import sys

file_path = "database/database_textdriven.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"❌ ERRORE CRITICO: Il file JSON è corrotto e non può essere letto. Dettagli: {e}")
    sys.exit(1)
except FileNotFoundError:
    print(f"❌ ERRORE: File {file_path} non trovato.")
    sys.exit(1)

concept_dict = data.get("concept_dict", {})
total_concepts = len(concept_dict)

print(f"✅ JSON caricato correttamente.")
print(f"📊 Totale concetti salvati: {total_concepts}")

# Chiavi che ci aspettiamo di trovare in ogni concetto
expected_keys = {"name", "image", "representative_image", "top_k_images", "info", "category"}

malformed_concepts = []

for concept_key, concept_data in concept_dict.items():
    missing_keys = expected_keys - set(concept_data.keys())
    
    if missing_keys:
        malformed_concepts.append(f"{concept_key} (Mancano chiavi: {missing_keys})")
    elif not isinstance(concept_data.get("info"), dict) or not concept_data["info"]:
        malformed_concepts.append(f"{concept_key} (Il campo 'info' è vuoto o non è un dizionario)")

print("-" * 40)
if malformed_concepts:
    print(f"⚠️ TROVATE ANOMALIE IN {len(malformed_concepts)} CONCETTI:")
    for issue in malformed_concepts:
        print(f"  - {issue}")
else:
    print("✅ Tutti i concetti salvati hanno la struttura corretta e completa.")