def stage_text_fix(database_path: str, output_dir: str) -> None:
    """Stage 3.5: Correzione testo con FluxText sui concept in graveyard."""
    print(f"\n{'='*70}\n✍️  STAGE: TEXT FIX (FluxText)\n{'='*70}")

    from pipeline.text_fix import (
        _extract_brand_texts, _detect_text_regions,
        _build_mask_from_regions, _build_prompt, call_fluxtext
    )

    recovery_path = os.path.join(output_dir, "recovery_results.json")
    if not os.path.exists(recovery_path):
        print("   ⚠️  recovery_results.json non trovato. Esegui prima lo stage recovery.")
        return

    with open(recovery_path, "r") as f:
        recovery_results = json.load(f)

    with open(database_path, "r") as f:
        database = json.load(f)
    concept_dict = database.get("concept_dict", {})

    # Filtra: graveyard + brand/text presente e nei failed attributes
    candidates = {}
    for concept_id, result in recovery_results.items():
        if result.get("status") != "unrecoverable":
            continue
        content = concept_dict.get(concept_id, {})
        fingerprints = content.get("info", {})
        brand_text = fingerprints.get("brand/text", "")
        if not brand_text or any(x in brand_text.lower() for x in ["none", "no visible", "no text"]):
            continue
        # Controlla che brand/text sia tra i failed
        failed = result.get("last_missing_details", [])
        if not any("brand" in f.lower() or "text" in f.lower() or "logo" in f.lower()
                   for f in failed):
            continue
        candidates[concept_id] = {
            "fingerprints": fingerprints,
            "brand_text": brand_text,
            "gen_image_path": os.path.join(output_dir, f"{concept_id}_generated.png"),
        }

    if not candidates:
        print("   ✅ Nessun concept candidato per text fix.")
        return

    print(f"   Trovati {len(candidates)} concept candidati per text fix.")

    FLUX_TEXT_URL = getattr(Config.Models, "FLUX_TEXT_URL", "http://127.0.0.1:8767")

    print("   Caricamento Qwen3-VL per estrazione testi...")
    reasoner = _build_reasoner()
    clip_calculator = ClipScoreCalculator(device="cuda")

    fixed_count = 0
    text_fix_report = {}

    for concept_id, data in candidates.items():
        print(f"\n   ✍️  {concept_id} | brand/text: {data['brand_text']}")

        gen_path = data["gen_image_path"]
        fingerprints = data["fingerprints"]

        if not os.path.exists(gen_path):
            print(f"      ⚠️  Immagine non trovata: {gen_path}")
            text_fix_report[concept_id] = {"status": "skipped", "reason": "Image not found"}
            continue

        # 1. Estrai testi puliti
        texts = _extract_brand_texts(reasoner, data["brand_text"])
        print(f"      📝 Testi estratti: {texts}")

        if not texts:
            print("      ⚠️  Nessun testo estratto, skip.")
            text_fix_report[concept_id] = {"status": "skipped", "reason": "No texts extracted"}
            continue

        # 2. Rileva zone testo con EasyOCR
        regions = _detect_text_regions(gen_path)
        print(f"      🔍 EasyOCR trovate {len(regions)} regioni testo")

        src_img = Image.open(gen_path).convert("RGB")
        img_w, img_h = src_img.size

        if not regions:
            # Nessun testo trovato — usa una region centrale come fallback
            print("      ⚠️  Nessuna regione trovata, uso fallback centrale.")
            regions = [{"bbox": [img_w//4, img_h//4, 3*img_w//4, img_h//2], "text": "", "conf": 0.0}]

        # 3. Costruisci mask
        mask = _build_mask_from_regions(regions, img_w, img_h)

        # 4. Salva la _generated corrente come _pretext prima di sovrascriverla
        import shutil
        pretext_path = os.path.join(output_dir, f"{concept_id}_generated_pretext.png")
        shutil.copy2(gen_path, pretext_path)

        # 5. Costruisci prompt e chiama FluxText
        prompt = _build_prompt(fingerprints, texts)
        print(f"      🚀 Prompt: {prompt[:80]}...")

        success = call_fluxtext(
            flux_text_url=FLUX_TEXT_URL,
            source_image_path=gen_path,
            mask=mask,
            texts=texts,
            prompt=prompt,
            seed=Config.Generate.SEED,
        )

        if success:
            # 6. Verifica risultato
            ref_path = _get_first_image(concept_dict.get(concept_id, {}))
            if ref_path:
                verification = verify_generation_r2p(
                    reasoner=reasoner,
                    clip_calculator=clip_calculator,
                    gen_image_path=gen_path,
                    ref_image_path=ref_path,
                    fingerprints=fingerprints,
                )
                is_verified = verification["is_verified"]
                score = verification["score"]
            else:
                is_verified = False
                score = 0.0

            if is_verified:
                print(f"      ✅ TEXT FIX RIUSCITO! Score: {score:.2f}")
                fixed_count += 1
                text_fix_report[concept_id] = {
                    "status": "fixed",
                    "score": score,
                    "texts_rendered": texts,
                    "pretext_image": pretext_path,
                }
                # Aggiorna recovery_results
                recovery_results[concept_id]["status"] = "recovered_text_fix"
                recovery_results[concept_id]["final_score"] = score
            else:
                print(f"      ❌ Verifica fallita dopo text fix. Score: {score:.2f}")
                # Ripristina immagine precedente
                shutil.copy2(pretext_path, gen_path)
                text_fix_report[concept_id] = {
                    "status": "failed_verification",
                    "score": score,
                    "texts_rendered": texts,
                }
        else:
            print(f"      ❌ FluxText generation fallita.")
            shutil.copy2(pretext_path, gen_path)
            text_fix_report[concept_id] = {"status": "failed_generation", "texts_rendered": texts}

    # Salva reports
    text_fix_path = os.path.join(output_dir, "text_fix_results.json")
    with open(text_fix_path, "w") as f:
        json.dump(text_fix_report, f, indent=4)

    # Aggiorna anche recovery_results con i nuovi status
    with open(recovery_path, "w") as f:
        json.dump(recovery_results, f, indent=4)

    print(f"\n📊 Text Fix: {fixed_count}/{len(candidates)} corretti → {text_fix_path}")

    del reasoner
    del clip_calculator
    cleanup_gpu()