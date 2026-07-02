#!/usr/bin/env python3
"""
Check progress of R2P-GEN pipeline.
"""
import os
import json
from pathlib import Path

def check_progress(database_path: str, output_dir: str):
    print("=" * 60)
    print("📊 R2P-GEN Progress Check")
    print("=" * 60)
    
    # Check database
    if os.path.exists(database_path):
        with open(database_path, 'r') as f:
            db = json.load(f)
        concepts = db.get("concept_dict", {})
        print(f"\n✅ Database: {database_path}")
        print(f"   Concepts: {len(concepts)}")
    else:
        print(f"\n❌ Database not found: {database_path}")
        return
    
    # Check generated images
    output_path = Path(output_dir)
    if output_path.exists():
        generated = list(output_path.glob("*_generated.png"))
        print(f"\n📁 Output: {output_dir}")
        print(f"   Generated images: {len(generated)} / {len(concepts)}")
        print(f"   Progress: {100*len(generated)/max(1,len(concepts)):.1f}%")
        
        # Find missing
        missing = []
        for concept_id in concepts.keys():
            img_path = output_path / f"{concept_id}_generated.png"
            if not img_path.exists():
                missing.append(concept_id)
        
        if missing:
            print(f"\n⚠️  Missing generations ({len(missing)}):")
            for m in missing[:10]:
                print(f"      - {m}")
            if len(missing) > 10:
                print(f"      ... and {len(missing)-10} more")
    else:
        print(f"\n❌ Output directory not found: {output_dir}")
    
    # Check verification results
    results_path = output_path / "verification_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        passed = sum(1 for r in results.values() if r.get("is_verified"))
        print(f"\n📋 Verification Results:")
        print(f"   Verified: {passed} / {len(results)}")
        print(f"   Pass rate: {100*passed/max(1,len(results)):.1f}%")
    else:
        print(f"\n⏳ Verification not yet run")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="database/database_test.json")
    parser.add_argument("--output", default="output/generated")
    args = parser.parse_args()
    
    check_progress(args.database, args.output)