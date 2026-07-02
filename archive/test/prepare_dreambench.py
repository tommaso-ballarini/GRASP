import os
import shutil
from pathlib import Path

# Percorsi di input e output
SOURCE_DIR = "/leonardo_work/IscrC_MUSE/tballari/FM_Data/dreambooth/dataset" # La cartella appena scaricata con git
TARGET_DIR = "/leonardo_work/IscrC_MUSE/tballari/FM_Data/dreambench-data/test" # Simula lo split "test" del tuo perva-data

def convert_dreambench_structure():
    source_path = Path(SOURCE_DIR)
    target_path = Path(TARGET_DIR)
    
    if not source_path.exists():
        print(f"Errore: La cartella {SOURCE_DIR} non esiste. Hai scaricato il repo?")
        return

    # Itera attraverso i 30 soggetti di DreamBench
    for subject_dir in source_path.iterdir():
        if subject_dir.is_dir() and not subject_dir.name.startswith('.'):
            subject_name = subject_dir.name
            
            # Creiamo la cartella target simulando category = subject_name e concept_id = "001"
            target_subject_dir = target_path / subject_name / "001"
            target_subject_dir.mkdir(parents=True, exist_ok=True)
            
            # Copiamo le immagini
            image_count = 0
            for img_file in subject_dir.glob("*.jpg"):
                # Rinominiamo o semplicemente copiamo
                dest_file = target_subject_dir / img_file.name
                shutil.copy2(img_file, dest_file)
                image_count += 1
                
            print(f"Mappato {subject_name}: {image_count} immagini copiate in {target_subject_dir}")

    print("\n✅ Conversione completata! Ora puoi puntare config.py alla cartella 'dreambench-data'")

if __name__ == "__main__":
    convert_dreambench_structure()