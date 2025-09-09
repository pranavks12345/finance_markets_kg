"""
Finance NER Post-Processor (Final Locked Version with PERSON normalization)
==========================================================================

Entity types: COMPANY, TICKER, MONEY, PERCENTAGE, DATE, ETF, INDEX, PERSON
"""

import os, re, json, unicodedata
from pathlib import Path
from datetime import datetime

INPUT_FILE = Path("evaluation_results/perfect_ner_results_final_one.json")
OUTPUT_DIR = Path("evaluation_results")

COMPANY_CANONICAL = {
    "apple": "Apple Inc",
    "microsoft": "Microsoft Corp",
    "amazon": "Amazon.com Inc",
    "alphabet": "Alphabet Inc",
    "google": "Alphabet Inc",
    "google llc": "Alphabet Inc",
    "meta": "Meta Platforms Inc",
    "meta platforms inc": "Meta Platforms Inc",
    "tesla": "Tesla Inc",
    "nvidia": "Nvidia Corp",
    "coca cola": "Coca-Cola",
    "coca-cola": "Coca-Cola",
    "exxonmobil": "ExxonMobil",
    "exxon mobil": "ExxonMobil",
    "jp morgan": "J.P. Morgan",
    "j.p. morgan": "J.P. Morgan",
    "pomerantz law firm": "Pomerantz LLP",
    "ishares msci usa quality garp etf": "iShares MSCI USA Quality GARP ETF",
}

CAPITALIZATION_FIXES = {
    "factset": "FactSet",
    "wsj": "WSJ",
    "idc": "IDC",
    "ubs": "UBS",
    "reuters": "Reuters",
    "ishares": "iShares",
}

ETF_TICKERS = {"IVE", "SPY", "SCHD", "QQQ", "IVV", "VOO", "DIA"}
INDEX_NAMES = {"s&p 500", "s & p 500", "dow jones", "nasdaq", "nasdaq 100"}

NOISE_TERMS = {"magnificent 7", "magnificent seven", "these", "shimmers,", "exclusive-us"}
JUNK_REGEX = re.compile(r"^[\W\d_]+$")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()

def normalize_money(text: str) -> str:
    return re.sub(r'(\$[\d,.]+)\s*(billion|million|trillion|bn|mn|tn)',
                  lambda m: f"{m.group(1)} {m.group(2).lower()}",
                  text.replace(" .", "."), flags=re.I)

def normalize_percent(text: str) -> str:
    t = text.replace(" .", ".")
    t = re.sub(r'(\d+)\s*%', r'\1%', t)
    return re.sub(r'(\d+)\s*percent', r'\1%', t, flags=re.I)

def normalize_date(text: str) -> str:
    return re.sub(r'\bQ([1-4])\s*[, ]*\s*(\d{4})', r'Q\1 \2', text.strip(), flags=re.I)

def canonicalize_company(name: str) -> str:
    name = re.sub(r"\s+\.", ".", name.strip())
    name = re.sub(r"\bInc\s*\.\b", "Inc", name, flags=re.I)
    name = re.sub(r"\bCorp\s*\.\b", "Corp", name, flags=re.I)
    key = name.lower().strip()
    if key in COMPANY_CANONICAL: return COMPANY_CANONICAL[key]
    if key in CAPITALIZATION_FIXES: return CAPITALIZATION_FIXES[key]
    return name

def normalize_ticker(ticker: str) -> str:
    return ticker.upper()

def normalize_person(name: str) -> str:
    """Normalize PERSON names with proper capitalization."""
    name = re.sub(r"\s+", " ", name.strip())
    name = re.sub(r"\bo\s*['’]\s*([a-z])", lambda m: "O'" + m.group(1).upper(), name, flags=re.I)
    parts = []
    for token in name.split(" "):
        if not token: continue
        if token.lower() in {"iii", "jr", "sr"}:
            parts.append(token.upper())
        else:
            parts.append(token[0].upper() + token[1:].lower())
    return " ".join(parts)

def classify_index_or_etf(entity, audit_log):
    txt, lbl = entity["text"], entity["label"]
    low = txt.lower()
    if lbl == "COMPANY":
        if txt.upper() in ETF_TICKERS or "etf" in low:
            entity["label"] = "ETF"
            audit_log["reclassified_etf_index"].append(f"{txt} COMPANY → ETF")
        elif low in INDEX_NAMES:
            entity["label"] = "INDEX"
            audit_log["reclassified_etf_index"].append(f"{txt} COMPANY → INDEX")
    return entity

def global_company_merge(all_articles, audit_log):
    global_map = {}
    for article in all_articles:
        for e in article.get("entities", []):
            if e["label"] == "COMPANY":
                txt = canonicalize_company(e["text"])
                key = re.sub(r'\b(inc|corp|corporation|ltd|llc|plc)\b\.?', '', txt, flags=re.I)
                key = re.sub(r'[^a-z0-9 ]+', '', key.lower()).strip()
                if not key: continue
                if key in COMPANY_CANONICAL: global_map[key] = COMPANY_CANONICAL[key]
                else:
                    if key not in global_map: global_map[key] = txt
                    elif len(txt) > len(global_map[key]): global_map[key] = txt
    for article in all_articles:
        for e in article.get("entities", []):
            if e["label"] == "COMPANY":
                txt = canonicalize_company(e["text"])
                key = re.sub(r'\b(inc|corp|corporation|ltd|llc|plc)\b\.?', '', txt, flags=re.I)
                key = re.sub(r'[^a-z0-9 ]+', '', key.lower()).strip()
                if key in global_map and txt != global_map[key]:
                    audit_log["alias_collapsed"].append(f"{txt} → {global_map[key]}")
                    e["text"] = global_map[key]
    return all_articles

def clean_entities(entities, audit_log):
    cleaned, seen = [], set()
    for e in entities:
        text, label = normalize_text(e.get("text", "")), e.get("label", "")
        if not text or not label: continue
        tl = text.lower()
        if tl in NOISE_TERMS or JUNK_REGEX.match(text) or len(text) < 2:
            audit_log["removed_noise"].append(f"{text} ({label})"); continue
        original = text
        if label == "MONEY": text = normalize_money(text)
        elif label == "PERCENTAGE": text = normalize_percent(text)
        elif label == "DATE": text = normalize_date(text)
        elif label == "COMPANY": text = canonicalize_company(text)
        elif label == "TICKER": text = normalize_ticker(text)
        elif label == "PERSON":
            new_text = normalize_person(text)
            if new_text != text:
                audit_log["normalized_person"].append(f"{text} → {new_text}")
            text = new_text
        if text != original and label != "PERSON":
            audit_log["normalized"].append(f"{original} → {text}")
        key = (label, text.lower())
        if key in seen:
            audit_log["deduplicated"].append(f"Duplicate removed: {text} ({label})"); continue
        seen.add(key)
        new_entity = classify_index_or_etf({"text": text, "label": label}, audit_log)
        cleaned.append(new_entity)
    return sorted(cleaned, key=lambda x: (x["label"], x["text"].lower()))

def has_key_entity(article, labels=("COMPANY", "TICKER", "MONEY")):
    return any(e["label"] in labels for e in article.get("entities", []))

def stage2_restore(article, audit_log):
    kept = []
    for e in article.get("entities", []):
        if e["label"] in {"DATE", "PERCENTAGE"} and not has_key_entity(article):
            audit_log["contextual_drops"].append(f"Dropped isolated {e['text']} ({e['label']})"); continue
        kept.append(e)
    article["entities"], article["entity_count"] = kept, len(kept)
    return article

def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}"); return
    print(f"Loading {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f: data = json.load(f)
    articles = list(data.values()) if isinstance(data, dict) else data
    if not isinstance(articles, list): raise ValueError("Unexpected JSON structure")
    stage1, final, dropped = [], [], 0
    audit_log = {"removed_noise": [], "deduplicated": [], "normalized": [],
                 "normalized_person": [], "contextual_drops": [],
                 "alias_collapsed": [], "reclassified_etf_index": []}
    for article in articles:
        if not isinstance(article, dict): continue
        if article.get("mode") == "forced_fallback": dropped += 1; continue
        article["entities"] = clean_entities(article.get("entities", []), audit_log)
        article["entity_count"] = len(article["entities"])
        stage1.append(article)
    for article in stage1: final.append(stage2_restore(article, audit_log))
    final = global_company_merge(final, audit_log)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"cleaned_results_stage1_{ts}.json","w",encoding="utf-8") as f: json.dump(stage1,f,indent=2,ensure_ascii=False)
    with open(OUTPUT_DIR / f"cleaned_results_final_one.json","w",encoding="utf-8") as f: json.dump(final,f,indent=2,ensure_ascii=False)
    with open(OUTPUT_DIR / f"audit_log_{ts}.json","w",encoding="utf-8") as f: json.dump({**audit_log,"dropped_forced_fallback": dropped},f,indent=2)
    print(f"\n==== SUMMARY ====\nArticles loaded: {len(articles)}\nForced_fallback dropped: {dropped}\nStage1 articles: {len(stage1)}\nFinal articles: {len(final)}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    main()

