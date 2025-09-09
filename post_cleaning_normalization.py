#!/usr/bin/env python3
"""
Final Production-Ready Dataset Cleanup
- Conservative approach focused on quality over quantity
- Aggressive noise filtering
- Standardized 6-label schema for consistency
- High-precision entity validation
"""

import json, re
from collections import Counter, defaultdict

class FinalCleanupProcessor:
    def __init__(self):
        # Core 6 labels only - everything else gets filtered
        self.valid_labels = {"COMPANY", "PERSON", "MONEY", "PERCENTAGE", "DATE", "TICKER"}
        
        # Company normalization - only major ones to avoid errors
        self.company_map = {
            "apple": "Apple Inc.", "apple inc": "Apple Inc.", "apple inc.": "Apple Inc.",
            "microsoft": "Microsoft Corporation", "microsoft corporation": "Microsoft Corporation",
            "nvidia": "NVIDIA Corporation", "nvidia corporation": "NVIDIA Corporation",
            "alphabet": "Alphabet Inc.", "google": "Alphabet Inc.",
            "meta": "Meta Platforms Inc.", "facebook": "Meta Platforms Inc.",
            "amazon": "Amazon.com Inc.", "amazon.com": "Amazon.com Inc.",
            "tesla": "Tesla Inc.", "tesla inc": "Tesla Inc.",
            "openai": "OpenAI Inc.",
            "jpmorgan": "JPMorgan Chase & Co.", "jp morgan": "JPMorgan Chase & Co.",
            "goldman sachs": "Goldman Sachs Group Inc.",
            "berkshire": "Berkshire Hathaway Inc.", "berkshire hathaway": "Berkshire Hathaway Inc.",
            "blackrock": "BlackRock Inc.",
            "bloomberg": "Bloomberg LP",
            "yahoo finance": "Yahoo Finance", "reuters": "Reuters"
        }
        
        # Known valid tickers only
        self.valid_tickers = {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA",
            "JPM", "GS", "MS", "BAC", "WFC", "C", "V", "MA", "BRK", "SPY", 
            "QQQ", "IWM", "VTI", "BND", "GLD", "TLT", "XLE", "XLF", "XLK"
        }
        
        # Noise words to completely filter out
        self.noise_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
            "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", 
            "with", "you", "your", "his", "her", "they", "them", "this", "these", "those",
            "but", "or", "not", "can", "could", "would", "should", "may", "might", "up",
            "out", "so", "if", "what", "how", "when", "where", "who", "why", "over", 
            "under", "above", "below", "before", "after", "during", "while", "about",
            "back", "down", "off", "all", "some", "more", "most", "many", "much", "few",
            "other", "another", "both", "each", "every", "any", "no", "none", "one", "two",
            "first", "last", "next", "new", "old", "big", "small", "good", "bad", "best",
            "said", "says", "told", "asked", "made", "make", "take", "get", "give", "go",
            "come", "know", "think", "see", "look", "want", "use", "work", "way", "day",
            "time", "year", "years", "month", "week", "today", "now", "then", "here", "there",
            # Financial noise
            "growth", "stock", "stocks", "market", "company", "companies", "business",
            "industry", "sector", "investment", "trading", "price", "prices", "value",
            "report", "reports", "news", "analysis", "data", "performance", "earnings",
            "revenue", "profit", "sales", "margin", "cost", "costs", "risk", "risks"
        }
        
        self.stats = Counter()

    def is_valid_company(self, text):
        """Conservative company validation"""
        text_clean = text.lower().strip()
        
        # Known companies
        if text_clean in self.company_map:
            return True
        
        # Must contain company indicators
        company_indicators = ["inc", "corp", "company", "ltd", "llc", "group", "fund", "bank", "capital"]
        if any(indicator in text_clean for indicator in company_indicators):
            return True
            
        # Known patterns like "S&P 500", "Magnificent Seven"
        if re.match(r'^[A-Z&\s]+\d+$', text) or "magnificent" in text_clean:
            return True
            
        return False

    def is_valid_person(self, text):
        """Conservative person validation"""
        # Must have first and last name
        words = text.strip().split()
        if len(words) < 2:
            return False
        
        # Each word should start with capital letter
        if not all(word[0].isupper() for word in words if word):
            return False
            
        return True

    def is_valid_ticker(self, text):
        """Only allow known valid tickers"""
        return text.upper() in self.valid_tickers

    def is_valid_money(self, text):
        """Strict money validation"""
        # Must have currency symbol or amount
        if not re.search(r'[\$€£¥]|\d', text):
            return False
            
        # Filter out partial amounts like "$6."
        if re.match(r'^\$?\d+\.?$', text) and len(text) < 4:
            return False
            
        return True

    def is_valid_percentage(self, text):
        """Strict percentage validation"""
        return bool(re.search(r'\d.*%|\d.*percent', text, re.I))

    def is_valid_date(self, text):
        """Strict date validation"""
        date_patterns = [
            r'\b20\d{2}\b',  # Years
            r'\bQ[1-4]\s*20\d{2}\b',  # Quarters
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s*20\d{2}',
            r'\d{1,2}[/-]\d{1,2}[/-]20\d{2}'
        ]
        return any(re.search(pattern, text, re.I) for pattern in date_patterns)

    def clean_entity(self, text, label):
        """Clean and validate individual entity"""
        text = text.strip()
        
        # Filter noise words immediately
        if text.lower() in self.noise_words:
            self.stats['noise_filtered'] += 1
            return None
            
        # Filter very short non-ticker entities
        if len(text) <= 2 and label != "TICKER":
            self.stats['too_short'] += 1
            return None
        
        # Label-specific validation and cleaning
        if label == "COMPANY":
            if not self.is_valid_company(text):
                self.stats['invalid_company'] += 1
                return None
            # Normalize known companies
            normalized = self.company_map.get(text.lower(), text)
            return normalized
            
        elif label == "PERSON":
            if not self.is_valid_person(text):
                self.stats['invalid_person'] += 1
                return None
                
        elif label == "TICKER":
            if not self.is_valid_ticker(text):
                self.stats['invalid_ticker'] += 1
                return None
            return text.upper()
            
        elif label == "MONEY":
            if not self.is_valid_money(text):
                self.stats['invalid_money'] += 1
                return None
                
        elif label == "PERCENTAGE":
            if not self.is_valid_percentage(text):
                self.stats['invalid_percentage'] += 1
                return None
                
        elif label == "DATE":
            if not self.is_valid_date(text):
                self.stats['invalid_date'] += 1
                return None
        
        return text

    def consolidate_label(self, original_label):
        """Map all variants to 6 core labels"""
        label = original_label.upper().strip()
        
        # Direct mapping
        label_map = {
            # Company variants
            "COMPANY": "COMPANY", "ORG": "COMPANY", "ORGANIZATION": "COMPANY",
            "PUBLISHER": "COMPANY", "INDEX": "COMPANY", "ALIAS": "COMPANY",
            "GOV_ORG": "COMPANY", "PRODUCT": "COMPANY",
            
            # Person variants
            "PERSON": "PERSON", "TITLE": "PERSON",
            
            # Financial variants
            "MONEY": "MONEY", "PERCENTAGE": "PERCENTAGE", "TICKER": "TICKER",
            
            # Date variants
            "DATE": "DATE", "TIME_PERIOD": "DATE", "EVENT": "DATE",
            
            # Everything else filtered out
            "CONCEPT": None, "POLICY": None, "LOCATION": None, "COUNTRY": None
        }
        
        result = label_map.get(label)
        if result is None:
            self.stats['labels_filtered'] += 1
        return result

    def process_article(self, article):
        """Process single article with aggressive filtering"""
        clean_entities = []
        
        original_entities = article.get("entities", [])
        if not original_entities:
            return {**article, "entities": []}
        
        for entity in original_entities:
            # Get clean label
            new_label = self.consolidate_label(entity.get("label", ""))
            if not new_label:
                continue
                
            # Get clean text
            clean_text = self.clean_entity(entity.get("text", ""), new_label)
            if not clean_text:
                continue
                
            clean_entities.append({"text": clean_text, "label": new_label})
            self.stats['entities_kept'] += 1
        
        # Remove exact duplicates
        seen = set()
        deduped = []
        for entity in clean_entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                deduped.append(entity)
            else:
                self.stats['duplicates_removed'] += 1
        
        # Limit to 8 entities max per article for quality
        if len(deduped) > 8:
            # Prioritize by label importance
            priority = {"MONEY": 0, "PERCENTAGE": 1, "COMPANY": 2, "PERSON": 3, "TICKER": 4, "DATE": 5}
            deduped.sort(key=lambda x: priority.get(x["label"], 10))
            deduped = deduped[:8]
            self.stats['articles_truncated'] += 1
        
        return {**article, "entities": deduped}

    def process_file(self, input_file, output_file):
        """Process entire dataset"""
        print(f"Loading {input_file}...")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} articles...")
        processed = []
        
        for i, article in enumerate(data):
            clean_article = self.process_article(article)
            processed.append(clean_article)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} articles")
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(processed, f, indent=2, ensure_ascii=False)
        
        # Stats
        print(f"\nProcessing complete! Saved to {output_file}")
        print(f"\nStatistics:")
        for stat, count in self.stats.most_common():
            print(f"  {stat}: {count:,}")
        
        # Final label distribution
        label_counts = Counter()
        total_entities = 0
        articles_with_entities = 0
        
        for article in processed:
            entities = article.get("entities", [])
            if entities:
                articles_with_entities += 1
            for entity in entities:
                label_counts[entity["label"]] += 1
                total_entities += 1
        
        print(f"\nFinal Results:")
        print(f"  Total articles: {len(processed):,}")
        print(f"  Articles with entities: {articles_with_entities:,} ({articles_with_entities/len(processed)*100:.1f}%)")
        print(f"  Total entities: {total_entities:,}")
        print(f"  Avg entities per article: {total_entities/len(processed):.1f}")
        
        print(f"\nLabel Distribution:")
        for label, count in label_counts.most_common():
            print(f"  {label}: {count:,} ({count/total_entities*100:.1f}%)")

if __name__ == "__main__":
    processor = FinalCleanupProcessor()
    
    # Process the broken dataset back to something usable
    processor.process_file(
        "production_gold_dataset_5k_clean_balanced.json",  # The original, not the broken one
        "production_gold_dataset_5k_final_clean.json"
    )