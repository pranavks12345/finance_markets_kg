#!/usr/bin/env python3
"""
Final Lightweight Cleaner for Gold Dataset
------------------------------------------
- Drops truncated MONEY fragments (e.g. "$6.", "$3.44")
- Removes duplicate COMPANY+TICKER combos
- Normalizes PERCENTAGE to "%"-based format
"""

import json
import re
from collections import defaultdict

def is_truncated_money(text: str) -> bool:
    """Detect truncated or incomplete money strings"""
    if re.match(r'^\$\d+\.$', text):  # e.g. "$6."
        return True
    if re.match(r'^\$\d+(\.\d+)?$', text):  # e.g. "$3.44"
        return True
    return False

def normalize_percentage(text: str) -> str:
    """Normalize percentage formats to 'X%'"""
    txt = text.strip().lower()
    if "percent" in txt:
        txt = re.sub(r'\s*percent(age)?( points?| bps)?', '%', txt)
    return txt.replace(" ", "")

def clean_article(article):
    seen = defaultdict(set)
    cleaned_entities = []
    
    for e in article.get("entities", []):
        label = e["label"].upper()
        text = e["text"].strip()
        
        # Drop truncated MONEY
        if label == "MONEY" and is_truncated_money(text):
            continue
        
        # Normalize percentage
        if label == "PERCENTAGE":
            text = normalize_percentage(text)
        
        # Dedup COMPANY+TICKER pairs
        key = (text.lower(), label)
        if key in seen[label]:
            continue
        seen[label].add(key)
        
        cleaned_entities.append({"text": text, "label": label})
    
    article["entities"] = cleaned_entities
    return article

def process_file(infile, outfile):
    with open(infile, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = [clean_article(a) for a in data]

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    print(f"✅ Final cleaned dataset saved to {outfile}")
    print(f"Articles processed: {len(processed)}")
    total_entities = sum(len(a["entities"]) for a in processed)
    print(f"Total entities: {total_entities}")
    print(f"Avg entities/article: {total_entities/len(processed):.2f}")

if __name__ == "__main__":
    process_file(
        "production_gold_dataset_5k_final_clean.json",
        "production_gold_dataset_5k_final_ready.json"
    )
