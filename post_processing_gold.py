#!/usr/bin/env python3
"""
High-Precision/Recall Post-Processor for Gold Dataset
- Canonicalizes companies/tickers/publishers
- Removes hallucinations + noise
- Cleans MONEY/PERCENTAGE
- Deduplicates + splits compound entities
- Regex backfill for missed entities
"""

import json, re
from collections import Counter

class GoldPostProcessor:
    def __init__(self):
        self.company_map = {
            "apple": "Apple Inc.", "apple inc": "Apple Inc.",
            "microsoft": "Microsoft Corporation", "msft": "Microsoft Corporation",
            "nvidia": "NVIDIA Corporation", "nvda": "NVIDIA Corporation",
            "alphabet": "Alphabet Inc.", "google": "Alphabet Inc.",
            "meta": "Meta Platforms Inc.", "facebook": "Meta Platforms Inc.",
            "amazon": "Amazon.com Inc.", "tesla": "Tesla Inc.",
            "jpmorgan": "JPMorgan Chase & Co.", "jp morgan": "JPMorgan Chase & Co.",
            "goldman sachs": "Goldman Sachs Group Inc.",
            "berkshire": "Berkshire Hathaway Inc."
        }
        self.publisher_map = {
            "wsj": "Wall Street Journal", "ft": "Financial Times",
            "nyt": "New York Times", "bloomberg": "Bloomberg LP",
            "yahoo finance": "Yahoo Finance", "reuters": "Reuters"
        }
        self.noise = {"growth","margin","bullish","stocks","industry","company","market","trend"}
        self.stats = Counter()

    def normalize_company(self, txt):
        return self.company_map.get(txt.lower().strip(), txt)

    def normalize_publisher(self, txt):
        return self.publisher_map.get(txt.lower().strip(), txt)

    def clean_money(self, txt):
        if not re.search(r'[\$€£]|USD|EUR|GBP', txt): return None
        if re.fullmatch(r'\$?\d+', txt): return None
        return txt

    def clean_percentage(self, txt):
        return txt if re.search(r'\d', txt) else None

    def dedup(self, ents):
        seen, out = set(), []
        for e in ents:
            k = (e["text"].lower(), e["label"])
            if k not in seen:
                seen.add(k); out.append(e)
        return out

    def split_compounds(self, ents):
        out = []
        for e in ents:
            m = re.match(r"^(.+?)\s*\(([A-Z]+:[A-Z]{1,5}|[A-Z]{1,5})\)$", e["text"])
            if m and e["label"] == "COMPANY":
                out.append({"text": self.normalize_company(m.group(1)), "label":"COMPANY"})
                out.append({"text": m.group(2), "label":"TICKER"})
            else:
                out.append(e)
        return out

    def regex_backfill(self, text, existing):
        existing_txt = {e["text"].lower() for e in existing}
        extras=[]
        for pat,lbl in [
            (r"\$[\d,.]+ ?(?:trillion|billion|million|B|M|K)?", "MONEY"),
            (r"\d+(?:\.\d+)?%|\d+(?:\.\d+)? (?:percent|basis points|bps)", "PERCENTAGE"),
            (r"\bQ[1-4]\b|\b20\d{2}\b", "TIME_PERIOD")
        ]:
            for m in re.finditer(pat,text,flags=re.I):
                if m.group().lower() not in existing_txt:
                    extras.append({"text":m.group().strip(),"label":lbl})
        return extras

    def process_article(self, art):
        context=(art.get("headline","")+" "+art.get("summary","")).lower()
        ents=[]
        for e in art.get("entities",[]):
            txt=e["text"].strip(); lbl=e["label"].upper()
            if txt.lower() not in context: continue  # drop hallucination
            if lbl=="COMPANY": txt=self.normalize_company(txt)
            if lbl=="PUBLISHER": txt=self.normalize_publisher(txt)
            if lbl=="MONEY": txt=self.clean_money(txt)
            if lbl=="PERCENTAGE": txt=self.clean_percentage(txt)
            if lbl=="CONCEPT" and txt.lower() in self.noise: continue
            if txt: ents.append({"text":txt,"label":lbl})
        ents=self.split_compounds(ents)
        ents+=self.regex_backfill(context,ents)
        return {**art,"entities":self.dedup(ents)}

    def process_file(self,infile,outfile):
        data=json.load(open(infile))
        out=[self.process_article(a) for a in data]
        json.dump(out,open(outfile,"w"),indent=2)
        print(f"Saved cleaned file {outfile} ({len(out)} articles)")

if __name__=="__main__":
    GoldPostProcessor().process_file(
        "production_gold_dataset_5k.json",
        "production_gold_dataset_5k_clean_balanced.json"
    )

