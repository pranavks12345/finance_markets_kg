"""
Complete Scalable Perfect NER Pipeline - All-in-One with FIXED CHECKPOINTS
=======================================================================
Your original pipeline + scalable batch processing with robust checkpoint handling
"""

import os
import re
import json
import torch
import random
import asyncio
import time
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from dataclasses import dataclass
import backoff

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "model_path": "./finbert-ner-fixed",
    "max_length": 512,
    "confidence_threshold": 0.25,
    "raw_data_dir": "data/raw/news",
    "sample_size": 50,
    "gazetteers": {
        "companies": "data/gazetteers/companies.txt",
        "people": "data/gazetteers/people.txt",
        "tickers": "data/gazetteers/stock_symbols.txt"
    },
    "llm_model": "gpt-4o-mini",
    "entity_types": ["COMPANY", "PERSON", "TICKER", "MONEY", "PERCENTAGE", "DATE"],
    "output_dir": "evaluation_results",
    "use_llm_validation": True,
    "min_entities_for_llm": 2,
}

MONEY_PATTERNS = [
    r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\s*(?:billion|trillion|million|bn|tn|mn|B|T|M)\b',
    r'\$\s*\d+(?:\.\d{1,2})?\s*(?:billion|trillion|million|bn|tn|mn|B|T|M)\b',
    r'\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b'
]

PERCENTAGE_PATTERNS = [
    r'\b\d{1,3}(?:\.\d{1,2})?%\b',
    r'\b\d{1,3}(?:\.\d{1,2})?\s*percent\b',
    r'\b\d{1,3}(?:\.\d{1,2})?\s*percentage\s*points?\b',
    r'\bup\s+\d{1,3}(?:\.\d{1,2})?%\b',
    r'\bdown\s+\d{1,3}(?:\.\d{1,2})?%\b',
    r'\bgained?\s+\d{1,3}(?:\.\d{1,2})?%\b',
    r'\bfell?\s+\d{1,3}(?:\.\d{1,2})?%\b',
    r'\brose\s+\d{1,3}(?:\.\d{1,2})?%\b',
    r'\bdropped\s+\d{1,3}(?:\.\d{1,2})?%\b'
]

DATE_PATTERNS = [
    r'\bQ[1-4]\s*20\d{2}\b',
    r'\b(?:20\d{2})\b',
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*20\d{2}\b',
    r'\b\d{1,2}/\d{1,2}/20\d{2}\b'
]

TICKER_PATTERN = r'\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b'

TICKER_MAP = {
    "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "meta": "META", "facebook": "META",
    "nvidia": "NVDA", "netflix": "NFLX", "disney": "DIS", "uber": "UBER",
    "spotify": "SPOT", "paypal": "PYPL", "adobe": "ADBE", "salesforce": "CRM",
    "oracle": "ORCL", "intel": "INTC", "cisco": "CSCO", "ibm": "IBM",
    "amd": "AMD", "broadcom": "AVGO", "qualcomm": "QCOM"
}

def load_gazetteers(paths: Dict[str, str]) -> Dict[str, Set[str]]:
    """Load gazetteer files for reference (not strict validation)"""
    gazetteers = {}
    for key, filepath in paths.items():
        entries = set()
        try:
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            entries.add(line.lower())
                logger.info(f"Loaded {len(entries)} {key} entries")
            else:
                logger.warning(f"Gazetteer file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error loading {key}: {e}")
        gazetteers[key] = entries
    return gazetteers

def normalize_entity_text(text: str, label: str) -> str:
    """Normalize entity text based on its type"""
    if not text:
        return ""
    
    text = text.strip()
    
    if label == "TICKER":
        return text.upper()
    elif label == "MONEY":
        text = re.sub(r'\$\s+', '$', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*,\s*$', '', text)
        text = re.sub(r'\.\s+(\d)', r'.\1', text)
        return text
    elif label == "PERCENTAGE":
        text = re.sub(r'\s+', ' ', text)
        return text
    elif label == "DATE":
        return text
    elif label in ["COMPANY", "PERSON"]:
        words = []
        for word in text.split():
            if word.lower() in ["inc", "inc.", "corp", "corp.", "llc", "ltd", "plc", "co", "co."]:
                words.append(word.upper() if word.endswith('.') else word.capitalize())
            else:
                words.append(word.capitalize())
        return ' '.join(words)
    
    return text

def is_valid_entity(entity: Dict, gazetteers: Dict[str, Set[str]]) -> bool:
    """Minimal validation - ALWAYS extract something"""
    text = entity.get("text", "").strip()
    label = entity.get("label", "")
    
    if not text or not label or len(text) < 1:
        return False
    
    text_lower = text.lower()
    
    universal_blacklist = {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
    if text_lower in universal_blacklist:
        return False
    
    if label == "PERSON":
        words = text.split()
        if len(words) < 2:
            return False
        if text_lower.startswith(('ceo ', 'cfo ', 'president ', 'chairman ')):
            return False
        return True
    
    elif label == "COMPANY":
        if len(text) < 2:
            return False
        return True
    
    elif label == "TICKER":
        return re.match(TICKER_PATTERN, text) is not None
    
    elif label == "MONEY":
        return '$' in text and any(c.isdigit() for c in text)
    
    elif label == "PERCENTAGE":
        return ('%' in text or 'percent' in text.lower()) and any(c.isdigit() for c in text)
    
    elif label == "DATE":
        return any(c.isdigit() for c in text)
    
    return True

def extract_structured_entities(text: str) -> List[Dict]:
    """Extract structured entities (MONEY, PERCENTAGE, DATE) using regex"""
    entities = []
    
    for pattern in MONEY_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            money_text = re.sub(r'\s+', ' ', match.group().strip())
            entities.append({"text": money_text, "label": "MONEY"})
    
    for pattern in PERCENTAGE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            entities.append({"text": match.group().strip(), "label": "PERCENTAGE"})
    
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            entities.append({"text": match.group().strip(), "label": "DATE"})
    
    ticker_in_parens = r'\((?:NASDAQ|NYSE|OTC|AMEX):\s*([A-Z]{2,5})\)|\(([A-Z]{2,5})\)'
    for match in re.finditer(ticker_in_parens, text):
        ticker = match.group(1) or match.group(2)
        if ticker:
            entities.append({"text": ticker, "label": "TICKER"})
    
    return entities

def add_derived_tickers(entities: List[Dict]) -> List[Dict]:
    """Add ticker symbols for known companies"""
    enhanced_entities = entities.copy()
    existing_tickers = {e["text"] for e in entities if e["label"] == "TICKER"}
    
    for entity in entities:
        if entity["label"] == "COMPANY":
            company_name = entity["text"].lower()
            first_word = company_name.split()[0] if company_name.split() else ""
            
            if first_word in TICKER_MAP:
                ticker = TICKER_MAP[first_word]
                if ticker not in existing_tickers:
                    enhanced_entities.append({"text": ticker, "label": "TICKER"})
                    existing_tickers.add(ticker)
    
    return enhanced_entities

def get_money_base_number(text: str) -> str:
    """Extract base number from money text for comparison"""
    number_match = re.search(r'[\d,]+\.?\d*', text)
    if not number_match:
        return text.lower()
    
    number_str = number_match.group().replace(',', '')
    try:
        base_number = float(number_str)
        unit_match = re.search(r'(billion|trillion|million|bn|tn|mn|b|t|m)\b', text.lower())
        if unit_match:
            unit = unit_match.group()
            unit_map = {'bn': 'billion', 'tn': 'trillion', 'mn': 'million', 
                       'b': 'billion', 't': 'trillion', 'm': 'million'}
            unit = unit_map.get(unit, unit)
            return f"{base_number}_{unit}"
        return str(base_number)
    except:
        return number_str

def get_date_base_form(text: str) -> str:
    """Extract normalized date form for comparison"""
    text_clean = text.lower().strip()
    text_clean = re.sub(r'[,\s]+$', '', text_clean)
    
    if re.match(r'q[1-4]\s*20\d{2}', text_clean):
        quarter_match = re.search(r'q([1-4])', text_clean)
        year_match = re.search(r'20\d{2}', text_clean)
        if quarter_match and year_match:
            return f"q{quarter_match.group(1)} {year_match.group()}"
    
    if re.match(r'20\d{2}$', text_clean):
        return text_clean
    
    if re.match(r'(january|february|march|april|may|june|july|august|september|october|november|december)', text_clean):
        return re.sub(r'[,\s]+', ' ', text_clean).strip()
    
    return text_clean

def validate_entity_exists_in_text(entity: Dict, original_text: str) -> bool:
    """Check if entity actually exists in the original text to prevent hallucination"""
    entity_text = entity.get("text", "").strip()
    if not entity_text:
        return False
    
    text_lower = original_text.lower()
    entity_lower = entity_text.lower()
    
    if entity_lower in text_lower:
        return True
    
    if entity.get("label") == "PERSON":
        name_parts = entity_lower.split()
        if len(name_parts) >= 2:
            return all(part in text_lower for part in name_parts if len(part) > 2)
    
    if entity.get("label") == "COMPANY":
        entity_base = re.sub(r'\b(inc\.?|corp\.?|llc|ltd\.?|co\.?|corporation|company)\b', '', entity_lower).strip()
        if entity_base and entity_base in text_lower:
            return True
    
    return False

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entities with improved money handling"""
    if not entities:
        return entities
    
    seen = set()
    final_entities = []
    
    for entity in entities:
        label = entity["label"]
        text = entity["text"]
        
        if label == "MONEY":
            base_number = get_money_base_number(text)
            
            duplicate_found = False
            for i, existing in enumerate(final_entities):
                if existing["label"] == "MONEY":
                    existing_base = get_money_base_number(existing["text"])
                    existing_num = existing_base.split('_')[0]
                    current_num = base_number.split('_')[0]
                    
                    if existing_num == current_num:
                        if len(text) > len(existing["text"]):
                            final_entities[i] = entity
                        duplicate_found = True
                        break
            
            if not duplicate_found:
                final_entities.append(entity)
                
        elif label == "DATE":
            base_form = get_date_base_form(text)
            
            duplicate_found = False
            for i, existing in enumerate(final_entities):
                if existing["label"] == "DATE":
                    existing_base = get_date_base_form(existing["text"])
                    if base_form == existing_base:
                        if len(text) > len(existing["text"]):
                            final_entities[i] = entity
                        duplicate_found = True
                        break
            
            if not duplicate_found:
                final_entities.append(entity)
        else:
            if label == "PERCENTAGE":
                key = re.sub(r'\s+', '', text.lower())
            elif label == "COMPANY":
                key = text.lower().strip()
                key = re.sub(r'\s*,\s*', ' ', key)
                key = re.sub(r'\s+', ' ', key)
                key = re.sub(r'\b(inc\.?|corp\.?|llc|ltd\.?|co\.?)\s*$', '', key).strip()
            else:
                key = text.lower().strip()
            
            lookup_key = (label, key)
            if lookup_key not in seen:
                final_entities.append(entity)
                seen.add(lookup_key)
    
    return final_entities

class PerfectNERPipeline:
    """Your original Perfect NER Pipeline - unchanged"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.client = None
        self.gazetteers = load_gazetteers(config["gazetteers"])
        self._load_model()
        if config.get("use_llm_validation", False):
            self._setup_llm()

    def _load_model(self):
        """Load the financial NER model"""
        try:
            model_path = Path(self.config["model_path"])
            if model_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_path, local_files_only=True).to(device)
                logger.info(f"Financial NER model loaded from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}")
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")

    def _setup_llm(self):
        """Setup OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                logger.info("LLM client initialized")
            else:
                logger.warning("OpenAI API key not found")
        except Exception as e:
            logger.warning(f"LLM setup failed: {e}")

    def extract_with_model(self, text: str) -> List[Dict]:
        """Extract entities using the financial NER model"""
        if not self.model or not self.tokenizer:
            return []
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True, 
                max_length=self.config["max_length"], 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                
                probs = torch.softmax(logits, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                predictions = torch.argmax(logits, dim=1)
                
                confident_predictions = []
                for i, (pred, conf) in enumerate(zip(predictions, max_probs)):
                    if conf >= self.config["confidence_threshold"]:
                        confident_predictions.append(pred.item())
                    else:
                        confident_predictions.append(0)
                
                predictions = confident_predictions
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            labels = [self.model.config.id2label[pred] for pred in predictions]
            
            entities = self._reconstruct_entities_from_bio(tokens, labels)
            logger.info(f"Financial model extracted {len(entities)} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Model extraction error: {e}")
            return []

    def _reconstruct_entities_from_bio(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """Reconstruct entities from BIO tagging"""
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if token.startswith('[') and token.endswith(']'):
                continue
                
            if label == 'O':
                if current_entity and current_tokens:
                    full_text = ""
                    for t in current_tokens:
                        if t.startswith('##'):
                            full_text += t[2:]
                        else:
                            if full_text and not full_text.endswith(' '):
                                full_text += " " + t
                            else:
                                full_text += t
                    
                    if full_text.strip():
                        entities.append({
                            "text": full_text.strip(),
                            "label": current_entity
                        })
                
                current_entity = None
                current_tokens = []
                
            else:
                if '-' in label:
                    bio_tag, entity_type = label.split('-', 1)
                else:
                    bio_tag, entity_type = 'B', label
                
                if bio_tag == 'B':
                    if current_entity and current_tokens:
                        full_text = ""
                        for t in current_tokens:
                            if t.startswith('##'):
                                full_text += t[2:]
                            else:
                                if full_text and not full_text.endswith(' '):
                                    full_text += " " + t
                                else:
                                    full_text += t
                        
                        if full_text.strip():
                            entities.append({
                                "text": full_text.strip(),
                                "label": current_entity
                            })
                    
                    current_entity = entity_type
                    current_tokens = [token]
                    
                elif bio_tag == 'I' and entity_type == current_entity and current_tokens:
                    current_tokens.append(token)
        
        if current_entity and current_tokens:
            full_text = ""
            for t in current_tokens:
                if t.startswith('##'):
                    full_text += t[2:]
                else:
                    if full_text and not full_text.endswith(' '):
                        full_text += " " + t
                    else:
                        full_text += t
            
            if full_text.strip():
                entities.append({
                    "text": full_text.strip(),
                    "label": current_entity
                })
        
        return entities

    def extract_fallback_entities(self, text: str) -> List[Dict]:
        """Aggressive fallback extraction when all else fails"""
        entities = []
        
        company_pattern = r'\b[A-Z][A-Za-z&\.\s]{2,30}(?:Inc\.?|Corp\.?|LLC|Ltd\.?|Co\.?|LP)?\b'
        for match in re.finditer(company_pattern, text):
            candidate = match.group().strip()
            if len(candidate) > 2 and candidate not in ['The', 'And', 'For', 'But']:
                entities.append({"text": candidate, "label": "COMPANY"})
        
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b'
        for match in re.finditer(person_pattern, text):
            candidate = match.group().strip()
            entities.append({"text": candidate, "label": "PERSON"})
        
        ticker_pattern = r'\b[A-Z]{2,5}\b'
        for match in re.finditer(ticker_pattern, text):
            candidate = match.group().strip()
            if candidate not in ['THE', 'AND', 'FOR', 'BUT', 'ALL', 'NEW']:
                entities.append({"text": candidate, "label": "TICKER"})
        
        year_pattern = r'\b20\d{2}\b'
        for match in re.finditer(year_pattern, text):
            entities.append({"text": match.group(), "label": "DATE"})
        
        return entities

    def llm_validate_fallback(self, regex_results: str, original_text: str) -> List[Dict]:
        """LLM validates regex fallback results to filter out nonsense"""
        if not self.client:
            return []
        
        prompt = f"""Review these entities extracted from financial text and keep only the valid ones.

ORIGINAL TEXT: {original_text[:400]}

EXTRACTED ENTITIES: {regex_results}

Rules:
- Keep only actual business entities as COMPANY (not geographic locations, generic terms, or nonsense phrases)
- Keep only real person names as PERSON (not job titles or generic phrases)
- Keep only valid stock symbols as TICKER
- Reject any entity that is a geographic location, country, city, or region
- Reject nonsensical phrases or sentence fragments

Return JSON array of ONLY the valid entities:
[{{"text": "Apple Inc.", "label": "COMPANY"}}]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": "Validate entities strictly. Reject geographic locations and nonsense."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return [ent for ent in result if isinstance(ent, dict) and 
                           ent.get("text") and ent.get("label") in self.config["entity_types"]]
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        if isinstance(result, list):
                            return [ent for ent in result if isinstance(ent, dict) and 
                                   ent.get("text") and ent.get("label") in self.config["entity_types"]]
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.warning(f"LLM fallback validation failed: {e}")
        
        return []

    def llm_validate_cpt(self, text: str) -> List[Dict]:
        """LLM validation specifically for COMPANY, PERSON, TICKER entities with stricter rules"""
        if not self.client:
            return []
        
        prompt = f"""Extract ALL entities from this financial text. Be comprehensive but STRICT about what qualifies.

TEXT: {text[:1200]}

STRICT RULES:

PERSON - Must be actual human individuals with real names:
✓ VALID: Tim Cook, Warren Buffett, Elon Musk, Janet Yellen, Jerome Powell, Mark Zuckerberg
✗ INVALID: Job titles (CEO, President), generic phrases (Big Beautiful Bill), descriptive terms (Market Rally, Potential AI Bubble Risks), abstract concepts, legislation names, geographic references

COMPANY - Must be actual business entities or corporations:
✓ VALID: Apple Inc., Tesla, Goldman Sachs, Berkshire Hathaway, JPMorgan Chase, Meta Platforms
✗ INVALID: Countries/regions (USA, UK, China, Europe), cities (New York, London), generic terms (Market, These, Important), government bodies (Federal Reserve, Treasury), market concepts (Magnificent Seven, Big Tech)

TICKER - Stock symbols only:
✓ VALID: AAPL, TSLA, GS, BRK.A, META
✗ INVALID: Common words, abbreviations that aren't stock symbols

CRITICAL REJECTIONS:
- Any job title or role description
- Any generic word, phrase, or concept  
- Any abstract idea or market terminology
- Any geographic location (countries, cities, regions)
- Any government institution or agency
- Any descriptive phrase that isn't an actual name

Only extract entities that are definitely real people names or business entities.

Return JSON array only:
[{{"text": "Apple Inc.", "label": "COMPANY"}}, {{"text": "Tim Cook", "label": "PERSON"}}]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": "Extract financial entities with extreme precision. Reject anything that isn't clearly a person name or business entity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return [ent for ent in result if isinstance(ent, dict) and 
                           ent.get("text") and ent.get("label") in ["COMPANY", "PERSON", "TICKER"]]
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        if isinstance(result, list):
                            return [ent for ent in result if isinstance(ent, dict) and 
                                   ent.get("text") and ent.get("label") in ["COMPANY", "PERSON", "TICKER"]]
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.warning(f"LLM CPT validation failed: {e}")
        
        return []

    def llm_full_backup(self, text: str) -> List[Dict]:
        """Full LLM extraction when other methods find few entities"""
        if not self.client:
            return []
        
        prompt = f"""Extract EVERY financial entity from this text. Be extremely thorough but PRECISE about what qualifies.

TEXT: {text[:1200]}

Find ALL that meet STRICT criteria:

COMPANY - Only actual business entities:
✓ VALID: Tesla, Goldman Sachs, JPMorgan, Meta Platforms, any real corporation or fund
✗ INVALID: Countries (USA, China), cities (New York), regions (Europe), government bodies (Federal Reserve), generic terms (Market, Big Tech)

PERSON - Only real individual human names:
✓ VALID: Elon Musk, Warren Buffett, Janet Yellen, any actual person's name
✗ INVALID: Job titles (CEO, President), generic phrases, descriptive terms, abstract concepts

TICKER - Stock symbols only:
✓ VALID: TSLA, GS, META, any real stock symbol
✗ INVALID: Common words, non-stock abbreviations

MONEY - Dollar amounts:
✓ VALID: $5B, $1.2 million, $500

PERCENTAGE - Percentages:  
✓ VALID: 15%, 3.5%

DATE - Years/quarters:
✓ VALID: 2025, Q2 2025

BE COMPREHENSIVE for valid entities but REJECT anything that doesn't clearly fit the strict criteria.

Return JSON array only:
[{{"text": "Tesla", "label": "COMPANY"}}]"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.config["llm_model"],
                messages=[
                    {"role": "system", "content": "Extract entities comprehensively but with strict validation. Only real names and business entities."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=700
            )
            
            content = response.choices[0].message.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            
            try:
                result = json.loads(content)
                if isinstance(result, list):
                    return [ent for ent in result if isinstance(ent, dict) and 
                           ent.get("text") and ent.get("label") in self.config["entity_types"]]
            except json.JSONDecodeError:
                json_match = re.search(r'\[.*?\]', content, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        if isinstance(result, list):
                            return [ent for ent in result if isinstance(ent, dict) and 
                                   ent.get("text") and ent.get("label") in self.config["entity_types"]]
                    except json.JSONDecodeError:
                        pass
                        
        except Exception as e:
            logger.warning(f"LLM full backup failed: {e}")
        
        return []

    def extract_entities(self, text: str) -> Tuple[List[Dict], str]:
        """Perfect extraction pipeline with AI-first validation"""
        
        model_entities = self.extract_with_model(text)
        structured_entities = extract_structured_entities(text)
        
        all_entities = model_entities + structured_entities
        validated_entities = []
        
        for entity in all_entities:
            normalized_text = normalize_entity_text(entity["text"], entity["label"])
            if normalized_text:
                normalized_entity = {"text": normalized_text, "label": entity["label"]}
                if is_valid_entity(normalized_entity, self.gazetteers):
                    validated_entities.append(normalized_entity)
        
        validated_entities = add_derived_tickers(validated_entities)
        validated_entities = deduplicate_entities(validated_entities)
        
        logger.info(f"Running LLM validation for COMPANY/PERSON/TICKER")
        llm_cpt_entities = self.llm_validate_cpt(text)
        
        for entity in llm_cpt_entities:
            if not validate_entity_exists_in_text(entity, text):
                logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                continue
                
            normalized_text = normalize_entity_text(entity["text"], entity["label"])
            if normalized_text:
                normalized_entity = {"text": normalized_text, "label": entity["label"]}
                if is_valid_entity(normalized_entity, self.gazetteers):
                    validated_entities.append(normalized_entity)
        
        validated_entities = deduplicate_entities(validated_entities)
        entity_count = len(validated_entities)
        
        logger.info(f"After model+structured+LLM_CPT: {entity_count} entities")
        
        if entity_count < self.config["min_entities_for_llm"]:
            logger.info(f"Only {entity_count} entities found, using full LLM backup")
            llm_full_entities = self.llm_full_backup(text)
            
            for entity in llm_full_entities:
                if not validate_entity_exists_in_text(entity, text):
                    logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                    continue
                    
                normalized_text = normalize_entity_text(entity["text"], entity["label"])
                if normalized_text:
                    normalized_entity = {"text": normalized_text, "label": entity["label"]}
                    if is_valid_entity(normalized_entity, self.gazetteers):
                        validated_entities.append(normalized_entity)
            
            validated_entities = deduplicate_entities(validated_entities)
            entity_count = len(validated_entities)
            
        if entity_count == 0:
            logger.info(f"No entities found, using aggressive regex fallback")
            fallback_entities = self.extract_fallback_entities(text)
            
            if fallback_entities:
                logger.info(f"LLM validating {len(fallback_entities)} regex fallback entities")
                
                fallback_text = ", ".join([f"{e['text']} ({e['label']})" for e in fallback_entities])
                
                llm_validated_fallback = self.llm_validate_fallback(fallback_text, text)
                
                for entity in llm_validated_fallback:
                    if not validate_entity_exists_in_text(entity, text):
                        logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                        continue
                        
                    normalized_text = normalize_entity_text(entity["text"], entity["label"])
                    if normalized_text:
                        normalized_entity = {"text": normalized_text, "label": entity["label"]}
                        if is_valid_entity(normalized_entity, self.gazetteers):
                            validated_entities.append(normalized_entity)
                
                validated_entities = deduplicate_entities(validated_entities)
                entity_count = len(validated_entities)
                
                if entity_count > 0:
                    return validated_entities, "llm_validated_regex"
        
        if entity_count == 0:
            logger.info(f"FORCING entity extraction - cannot return empty")
            
            aggressive_llm = self.llm_full_backup(text)
            if aggressive_llm:
                for entity in aggressive_llm:
                    normalized_text = normalize_entity_text(entity["text"], entity["label"])
                    if normalized_text and len(normalized_text) > 1:
                        validated_entities.append({"text": normalized_text, "label": entity["label"]})
                        break
            
            if not validated_entities:
                headline_words = text.split()[:10]
                for word in headline_words:
                    if len(word) > 3 and word[0].isupper():
                        validated_entities.append({"text": word, "label": "COMPANY"})
                        break
                
                if not validated_entities:
                    validated_entities.append({"text": "Market", "label": "COMPANY"})
            
            return validated_entities, "forced_fallback"
        
        if entity_count < self.config["min_entities_for_llm"]:
            return validated_entities, "llm_backup"
        else:
            return validated_entities, "ai_validated"


@dataclass
class ScalingConfig:
    """Configuration for scalable processing"""
    batch_size: int = 20
    max_concurrent_batches: int = 2
    rate_limit_requests_per_minute: int = 350
    rate_limit_tokens_per_minute: int = 180000
    max_retries: int = 3
    retry_delay: float = 1.0
    max_articles: Optional[int] = None
    save_checkpoint_every: int = 50
    resume_from_checkpoint: bool = True
    output_dir: str = "evaluation_results"

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self.lock = threading.Lock()
        
    def can_make_request(self, estimated_tokens: int = 1000) -> Tuple[bool, float]:
        """Check if we can make a request and return wait time if not"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            if len(self.request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self.request_times[0])
                return False, wait_time
            
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                if self.token_usage:
                    wait_time = 60 - (now - self.token_usage[0][0])
                    return False, wait_time
            
            return True, 0.0
    
    def record_request(self, tokens_used: int):
        """Record a successful request"""
        with self.lock:
            now = time.time()
            self.request_times.append(now)
            self.token_usage.append((now, tokens_used))

class CheckpointManager:
    """FIXED checkpoint manager with success/failure tracking"""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "processing_checkpoint.json"
        self.results_file = self.output_dir / "partial_results.json"
        
    def save_checkpoint(self, successfully_processed: Set[int], failed_indices: Set[int], 
                       results: List[Dict], total_articles: int, batch_info: Dict):
        """Save checkpoint with success/failure tracking"""
        checkpoint_data = {
            "successfully_processed": list(successfully_processed),
            "failed_indices": list(failed_indices),
            "results_count": len(results),
            "total_articles": total_articles,
            "batch_info": batch_info,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        if results:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved: {len(successfully_processed)} successful, "
                   f"{len(failed_indices)} failed, {total_articles - len(successfully_processed) - len(failed_indices)} remaining")
    
    def load_checkpoint(self) -> Optional[Tuple[Set[int], Set[int], List[Dict]]]:
        """Load checkpoint and return (successful_indices, failed_indices, results)"""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            
            successfully_processed = set(checkpoint_data.get("successfully_processed", []))
            failed_indices = set(checkpoint_data.get("failed_indices", []))
            
            results = []
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            
            return successfully_processed, failed_indices, results
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Remove checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.results_file.exists():
            self.results_file.unlink()

class AsyncPipelineWrapper:
    """Async wrapper for the original pipeline with rate limiting"""
    
    def __init__(self, original_pipeline: PerfectNERPipeline, rate_limiter: RateLimiter):
        self.pipeline = original_pipeline
        self.rate_limiter = rate_limiter
        
        if self.pipeline.client:
            api_key = os.getenv("OPENAI_API_KEY")
            self.async_client = AsyncOpenAI(api_key=api_key) if api_key else None
        else:
            self.async_client = None
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _rate_limited_llm_call(self, messages: List[Dict], max_tokens: int = 600) -> Optional[str]:
        """Make rate-limited LLM request"""
        if not self.async_client:
            return None
            
        estimated_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4 + max_tokens
        
        can_proceed, wait_time = self.rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            logger.info(f"Rate limit hit, waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        try:
            response = await self.async_client.chat.completions.create(
                model=self.pipeline.config["llm_model"],
                messages=messages,
                temperature=0,
                max_tokens=max_tokens
            )
            
            self.rate_limiter.record_request(response.usage.total_tokens)
            content = response.choices[0].message.content.strip()
            return content.replace('```json', '').replace('```', '').strip()
            
        except Exception as e:
            logger.warning(f"Async LLM request failed: {e}")
            raise
    
    async def async_llm_validate_cpt(self, text: str) -> List[Dict]:
        """Async version of LLM validation using original prompts"""
        if not self.async_client:
            return []
        
        prompt = f"""Extract ALL entities from this financial text. Be comprehensive but STRICT about what qualifies.

TEXT: {text[:1200]}

STRICT RULES:

PERSON - Must be actual human individuals with real names:
✓ VALID: Tim Cook, Warren Buffett, Elon Musk, Janet Yellen, Jerome Powell, Mark Zuckerberg
✗ INVALID: Job titles (CEO, President), generic phrases (Big Beautiful Bill), descriptive terms (Market Rally, Potential AI Bubble Risks), abstract concepts, legislation names, geographic references

COMPANY - Must be actual business entities or corporations:
✓ VALID: Apple Inc., Tesla, Goldman Sachs, Berkshire Hathaway, JPMorgan Chase, Meta Platforms
✗ INVALID: Countries/regions (USA, UK, China, Europe), cities (New York, London), generic terms (Market, These, Important), government bodies (Federal Reserve, Treasury), market concepts (Magnificent Seven, Big Tech)

TICKER - Stock symbols only:
✓ VALID: AAPL, TSLA, GS, BRK.A, META
✗ INVALID: Common words, abbreviations that aren't stock symbols

CRITICAL REJECTIONS:
- Any job title or role description
- Any generic word, phrase, or concept  
- Any abstract idea or market terminology
- Any geographic location (countries, cities, regions)
- Any government institution or agency
- Any descriptive phrase that isn't an actual name

Only extract entities that are definitely real people names or business entities.

Return JSON array only:
[{{"text": "Apple Inc.", "label": "COMPANY"}}, {{"text": "Tim Cook", "label": "PERSON"}}]"""
        
        messages = [
            {"role": "system", "content": "Extract financial entities with extreme precision. Reject anything that isn't clearly a person name or business entity."},
            {"role": "user", "content": prompt}
        ]
        
        content = await self._rate_limited_llm_call(messages, 600)
        if not content:
            return []
        
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [ent for ent in result if isinstance(ent, dict) and 
                       ent.get("text") and ent.get("label") in ["COMPANY", "PERSON", "TICKER"]]
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if isinstance(result, list):
                        return [ent for ent in result if isinstance(ent, dict) and 
                               ent.get("text") and ent.get("label") in ["COMPANY", "PERSON", "TICKER"]]
                except json.JSONDecodeError:
                    pass
        
        return []
    
    async def async_llm_full_backup(self, text: str) -> List[Dict]:
        """Async version of full LLM backup using original prompts"""
        if not self.async_client:
            return []
        
        prompt = f"""Extract EVERY financial entity from this text. Be extremely thorough but PRECISE about what qualifies.

TEXT: {text[:1200]}

Find ALL that meet STRICT criteria:

COMPANY - Only actual business entities:
✓ VALID: Tesla, Goldman Sachs, JPMorgan, Meta Platforms, any real corporation or fund
✗ INVALID: Countries (USA, China), cities (New York), regions (Europe), government bodies (Federal Reserve), generic terms (Market, Big Tech)

PERSON - Only real individual human names:
✓ VALID: Elon Musk, Warren Buffett, Janet Yellen, any actual person's name
✗ INVALID: Job titles (CEO, President), generic phrases, descriptive terms, abstract concepts

TICKER - Stock symbols only:
✓ VALID: TSLA, GS, META, any real stock symbol
✗ INVALID: Common words, non-stock abbreviations

MONEY - Dollar amounts:
✓ VALID: $5B, $1.2 million, $500

PERCENTAGE - Percentages:  
✓ VALID: 15%, 3.5%

DATE - Years/quarters:
✓ VALID: 2025, Q2 2025

BE COMPREHENSIVE for valid entities but REJECT anything that doesn't clearly fit the strict criteria.

Return JSON array only:
[{{"text": "Tesla", "label": "COMPANY"}}]"""
        
        messages = [
            {"role": "system", "content": "Extract entities comprehensively but with strict validation. Only real names and business entities."},
            {"role": "user", "content": prompt}
        ]
        
        content = await self._rate_limited_llm_call(messages, 700)
        if not content:
            return []
        
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [ent for ent in result if isinstance(ent, dict) and 
                       ent.get("text") and ent.get("label") in self.pipeline.config["entity_types"]]
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if isinstance(result, list):
                        return [ent for ent in result if isinstance(ent, dict) and 
                               ent.get("text") and ent.get("label") in self.pipeline.config["entity_types"]]
                except json.JSONDecodeError:
                    pass
        
        return []

    async def async_llm_validate_fallback(self, regex_results: str, original_text: str) -> List[Dict]:
        """Async version of fallback validation using original prompts"""
        if not self.async_client:
            return []
        
        prompt = f"""Review these entities extracted from financial text and keep only the valid ones.

ORIGINAL TEXT: {original_text[:400]}

EXTRACTED ENTITIES: {regex_results}

Rules:
- Keep only actual business entities as COMPANY (not geographic locations, generic terms, or nonsense phrases)
- Keep only real person names as PERSON (not job titles or generic phrases)
- Keep only valid stock symbols as TICKER
- Reject any entity that is a geographic location, country, city, or region
- Reject nonsensical phrases or sentence fragments

Return JSON array of ONLY the valid entities:
[{{"text": "Apple Inc.", "label": "COMPANY"}}]"""
        
        messages = [
            {"role": "system", "content": "Validate entities strictly. Reject geographic locations and nonsense."},
            {"role": "user", "content": prompt}
        ]
        
        content = await self._rate_limited_llm_call(messages, 400)
        if not content:
            return []
        
        try:
            result = json.loads(content)
            if isinstance(result, list):
                return [ent for ent in result if isinstance(ent, dict) and 
                       ent.get("text") and ent.get("label") in self.pipeline.config["entity_types"]]
        except json.JSONDecodeError:
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    if isinstance(result, list):
                        return [ent for ent in result if isinstance(ent, dict) and 
                               ent.get("text") and ent.get("label") in self.pipeline.config["entity_types"]]
                except json.JSONDecodeError:
                    pass
        
        return []

    async def process_article_async(self, article: Dict, article_idx: int) -> Tuple[int, Dict]:
        """Process single article with async LLM calls - preserves your exact 5-stage logic"""
        logger.info(f"Processing article {article_idx}")
        
        text = f"{article['headline']} {article.get('summary', '')}"
        
        model_entities = self.pipeline.extract_with_model(text)
        structured_entities = extract_structured_entities(text)
        
        all_entities = model_entities + structured_entities
        validated_entities = []
        
        for entity in all_entities:
            normalized_text = normalize_entity_text(entity["text"], entity["label"])
            if normalized_text:
                normalized_entity = {"text": normalized_text, "label": entity["label"]}
                if is_valid_entity(normalized_entity, self.pipeline.gazetteers):
                    validated_entities.append(normalized_entity)
        
        validated_entities = add_derived_tickers(validated_entities)
        validated_entities = deduplicate_entities(validated_entities)
        
        llm_cpt_entities = await self.async_llm_validate_cpt(text)
        
        for entity in llm_cpt_entities:
            if not validate_entity_exists_in_text(entity, text):
                logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                continue
                
            normalized_text = normalize_entity_text(entity["text"], entity["label"])
            if normalized_text:
                normalized_entity = {"text": normalized_text, "label": entity["label"]}
                if is_valid_entity(normalized_entity, self.pipeline.gazetteers):
                    validated_entities.append(normalized_entity)
        
        validated_entities = deduplicate_entities(validated_entities)
        entity_count = len(validated_entities)
        
        if entity_count < self.pipeline.config["min_entities_for_llm"]:
            logger.info(f"Only {entity_count} entities found for article {article_idx}, using full LLM backup")
            llm_full_entities = await self.async_llm_full_backup(text)
            
            for entity in llm_full_entities:
                if not validate_entity_exists_in_text(entity, text):
                    logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                    continue
                    
                normalized_text = normalize_entity_text(entity["text"], entity["label"])
                if normalized_text:
                    normalized_entity = {"text": normalized_text, "label": entity["label"]}
                    if is_valid_entity(normalized_entity, self.pipeline.gazetteers):
                        validated_entities.append(normalized_entity)
            
            validated_entities = deduplicate_entities(validated_entities)
            entity_count = len(validated_entities)
        
        if entity_count == 0:
            logger.info(f"No entities found for article {article_idx}, using aggressive regex fallback")
            fallback_entities = self.pipeline.extract_fallback_entities(text)
            
            if fallback_entities:
                fallback_text = ", ".join([f"{e['text']} ({e['label']})" for e in fallback_entities])
                llm_validated_fallback = await self.async_llm_validate_fallback(fallback_text, text)
                
                for entity in llm_validated_fallback:
                    if not validate_entity_exists_in_text(entity, text):
                        logger.warning(f"Skipping potentially hallucinated entity: {entity['text']} ({entity['label']})")
                        continue
                        
                    normalized_text = normalize_entity_text(entity["text"], entity["label"])
                    if normalized_text:
                        normalized_entity = {"text": normalized_text, "label": entity["label"]}
                        if is_valid_entity(normalized_entity, self.pipeline.gazetteers):
                            validated_entities.append(normalized_entity)
                
                validated_entities = deduplicate_entities(validated_entities)
                entity_count = len(validated_entities)
        
        if entity_count == 0:
            logger.info(f"FORCING entity extraction for article {article_idx}")
            aggressive_llm = await self.async_llm_full_backup(text)
            if aggressive_llm:
                for entity in aggressive_llm:
                    normalized_text = normalize_entity_text(entity["text"], entity["label"])
                    if normalized_text and len(normalized_text) > 1:
                        validated_entities.append({"text": normalized_text, "label": entity["label"]})
                        break
            
            if not validated_entities:
                headline_words = text.split()[:10]
                for word in headline_words:
                    if len(word) > 3 and word[0].isupper():
                        validated_entities.append({"text": word, "label": "COMPANY"})
                        break
                
                if not validated_entities:
                    validated_entities.append({"text": "Market", "label": "COMPANY"})
        
        mode = "ai_validated"
        if entity_count < self.pipeline.config["min_entities_for_llm"]:
            mode = "llm_backup" if entity_count > 0 else "forced_fallback"
        
        result = {
            "headline": article["headline"],
            "summary": article.get("summary", ""),
            "entities": validated_entities,
            "entity_count": len(validated_entities),
            "mode": mode
        }
        
        if validated_entities:
            entity_summary = [f"{e['text']} ({e['label']})" for e in validated_entities[:5]]
            logger.info(f"Article {article_idx} found {len(validated_entities)} entities: {', '.join(entity_summary)}")
        else:
            logger.info(f"Article {article_idx} found no entities")
        
        return article_idx, result

def setup_system_persistence():
    """Configure macOS to prevent sleep/hibernation during processing"""
    import platform
    import subprocess
    
    system = platform.system().lower()
    
    if system == "darwin":
        try:
            caffeinate_process = subprocess.Popen([
                "caffeinate", "-d", "-i", "-s"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logger.info("macOS sleep prevention enabled (caffeinate)")
            return caffeinate_process
        except Exception as e:
            logger.warning(f"Could not prevent macOS sleep: {e}")
            logger.info("You may want to adjust Energy Saver settings manually")
    else:
        logger.warning(f"Sleep prevention not configured for {system}")
    
    return None

def setup_logging_for_overnight():
    """Setup comprehensive logging for overnight processing"""
    import logging.handlers
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"ner_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=100*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    return log_file

def setup_macos_energy_settings():
    """Provide instructions for macOS energy settings"""
    logger.info("\n" + "="*60)
    logger.info("macOS ENERGY SETTINGS RECOMMENDATIONS")
    logger.info("="*60)
    logger.info("For overnight processing, please check these settings:")
    logger.info("")
    logger.info("1. System Preferences > Energy Saver (or Battery on newer macOS):")
    logger.info("   - Set 'Turn display off after' to Never (or a long time)")
    logger.info("   - Uncheck 'Put hard disks to sleep when possible'")
    logger.info("   - Check 'Prevent computer from sleeping automatically'")
    logger.info("")
    logger.info("2. If on battery power:")
    logger.info("   - Connect power adapter for overnight processing")
    logger.info("   - Battery processing will drain power quickly")
    logger.info("")
    logger.info("3. Optional - Disable automatic software updates during processing")
    logger.info("="*60 + "\n")

def create_startup_script():
    """Create macOS-specific startup script for auto-resume"""
    
    script_dir = Path("scripts")
    script_dir.mkdir(exist_ok=True)
    
    shell_content = f"""#!/bin/bash

echo "Starting NER Pipeline..."
echo "Working directory: {Path.cwd()}"

cd "{Path.cwd()}"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3."
    exit 1
fi

if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting pipeline at $(date)"
python3 {__file__}

echo "Pipeline finished at $(date)"
echo "Press any key to close..."
read -n 1
"""
    
    script_path = script_dir / "run_pipeline.sh"
    with open(script_path, "w") as f:
        f.write(shell_content)
    os.chmod(script_path, 0o755)
    
    applescript_content = f'''
tell application "Terminal"
    do script "cd '{Path.cwd()}' && chmod +x scripts/run_pipeline.sh && ./scripts/run_pipeline.sh"
    activate
end tell
'''
    
    applescript_path = script_dir / "run_pipeline.scpt"
    with open(applescript_path, "w") as f:
        f.write(applescript_content)
    
    logger.info("Created macOS startup scripts:")
    logger.info(f"- Shell script: {script_path}")
    logger.info(f"- AppleScript: {applescript_path}")
    logger.info("You can double-click the .scpt file to run")

class ProcessMonitor:
    """Monitor processing and send notifications"""
    
    def __init__(self, total_articles: int):
        self.total_articles = total_articles
        self.start_time = time.time()
        self.last_progress_time = time.time()
        
    def log_progress(self, completed: int, current_cost_estimate: float):
        """Log detailed progress information"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        progress_pct = (completed / self.total_articles) * 100
        articles_per_hour = completed / (elapsed / 3600) if elapsed > 0 else 0
        
        eta_hours = (self.total_articles - completed) / articles_per_hour if articles_per_hour > 0 else 0
        eta_time = datetime.now() + timedelta(hours=eta_hours)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROGRESS UPDATE")
        logger.info(f"{'='*60}")
        logger.info(f"Completed: {completed:,}/{self.total_articles:,} ({progress_pct:.1f}%)")
        logger.info(f"Processing rate: {articles_per_hour:.1f} articles/hour")
        logger.info(f"Elapsed time: {elapsed/3600:.1f} hours")
        logger.info(f"Estimated completion: {eta_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Estimated cost so far: ${current_cost_estimate:.2f}")
        logger.info(f"{'='*60}\n")
        
        self.last_progress_time = current_time

class ScalablePipelineProcessor:
    """Enhanced processor with robust error handling and retry logic"""
    
    def __init__(self, original_pipeline: PerfectNERPipeline, scaling_config: ScalingConfig):
        self.pipeline = original_pipeline
        self.config = scaling_config
        self.rate_limiter = RateLimiter(
            scaling_config.rate_limit_requests_per_minute,
            scaling_config.rate_limit_tokens_per_minute
        )
        self.async_wrapper = AsyncPipelineWrapper(original_pipeline, self.rate_limiter)
        self.checkpoint_manager = CheckpointManager(scaling_config.output_dir)
    
    async def process_single_article_with_retry(self, article_idx: int, article: Dict, 
                                               max_retries: int = 3) -> Tuple[int, Optional[Dict], bool]:
        """Process single article with retry logic. Returns (index, result, success)"""
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Processing article {article_idx}, attempt {attempt + 1}")
                result = await self.async_wrapper.process_article_async(article, article_idx)
                return result[0], result[1], True
                
            except asyncio.TimeoutError:
                wait_time = min(30, 2 ** attempt)
                logger.warning(f"Article {article_idx} timeout (attempt {attempt + 1}), waiting {wait_time}s")
                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
                    
            except Exception as e:
                wait_time = min(60, 5 * (2 ** attempt))
                logger.warning(f"Article {article_idx} error (attempt {attempt + 1}): {e}")
                
                if "rate" in str(e).lower() or "429" in str(e):
                    wait_time = min(300, 60 * (2 ** attempt))
                    logger.info(f"Rate limit detected, waiting {wait_time}s before retry")
                
                if attempt < max_retries:
                    await asyncio.sleep(wait_time)
        
        logger.error(f"Article {article_idx} failed after {max_retries + 1} attempts")
        return article_idx, None, False
    
    async def process_batch_robust(self, batch_articles: List[Tuple[int, Dict]]) -> Tuple[List[Tuple[int, Dict]], Set[int]]:
        """Process batch with individual article retry logic"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        
        async def process_with_semaphore_and_retry(article_data):
            async with semaphore:
                article_idx, article = article_data
                return await self.process_single_article_with_retry(article_idx, article)
        
        tasks = [process_with_semaphore_and_retry(article_data) for article_data in batch_articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_results = []
        failed_indices = set()
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected batch error: {result}")
                failed_indices.update(idx for idx, _ in batch_articles)
            else:
                article_idx, article_result, success = result
                if success:
                    successful_results.append((article_idx, article_result))
                else:
                    failed_indices.add(article_idx)
        
        return successful_results, failed_indices
    
    async def retry_failed_articles(self, failed_indices: Set[int], articles: List[Dict]) -> Tuple[List[Dict], Set[int]]:
        """Retry failed articles individually with more aggressive retry logic"""
        if not failed_indices:
            return [], set()
            
        logger.info(f"Retrying {len(failed_indices)} failed articles individually")
        
        retry_results = []
        still_failed = set()
        
        for article_idx in sorted(failed_indices):
            if article_idx >= len(articles):
                continue
                
            idx, result, success = await self.process_single_article_with_retry(
                article_idx, articles[article_idx], max_retries=5
            )
            
            if success:
                retry_results.append(result)
                logger.info(f"Successfully retried article {article_idx}")
            else:
                still_failed.add(article_idx)
                logger.warning(f"Article {article_idx} still failing after individual retry")
        
        return retry_results, still_failed
    
    async def process_all_articles(self) -> List[Dict]:
        """Process all articles with robust checkpoint handling"""
        articles = self.load_articles()
        if not articles:
            logger.error("No articles found")
            return []
        
        total_articles = len(articles)
        if self.config.max_articles:
            articles = articles[:self.config.max_articles]
            total_articles = len(articles)
        
        logger.info(f"Processing {total_articles} articles with robust error handling")
        
        monitor = ProcessMonitor(total_articles)
        
        successfully_processed = set()
        failed_indices = set()
        all_results = []
        
        if self.config.resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_checkpoint()
            if checkpoint_data:
                successfully_processed, failed_indices, all_results = checkpoint_data
                logger.info(f"Resuming: {len(successfully_processed)} successful, "
                           f"{len(failed_indices)} previously failed")
        
        if failed_indices:
            logger.info(f"Retrying {len(failed_indices)} previously failed articles")
            retry_results, still_failed = await self.retry_failed_articles(failed_indices, articles)
            
            for result in retry_results:
                for idx, article in enumerate(articles):
                    if article["headline"] == result["headline"]:
                        successfully_processed.add(idx)
                        all_results.append(result)
                        failed_indices.discard(idx)
                        break
            
            failed_indices = still_failed
        
        remaining_indices = []
        for i in range(total_articles):
            if i not in successfully_processed and i not in failed_indices:
                remaining_indices.append(i)
        
        remaining_articles = [(i, articles[i]) for i in remaining_indices]
        
        if not remaining_articles:
            logger.info(f"Processing complete! {len(successfully_processed)} successful, "
                       f"{len(failed_indices)} failed")
            return all_results
        
        logger.info(f"Processing {len(remaining_articles)} remaining articles")
        
        for batch_start in range(0, len(remaining_articles), self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, len(remaining_articles))
            batch_articles = remaining_articles[batch_start:batch_end]
            
            batch_num = (batch_start // self.config.batch_size) + 1
            logger.info(f"Processing batch {batch_num}: articles {batch_start + 1}-{batch_end}")
            
            batch_results, batch_failed = await self.process_batch_robust(batch_articles)
            
            for article_idx, result in batch_results:
                successfully_processed.add(article_idx)
                all_results.append(result)
            
            failed_indices.update(batch_failed)
            
            completed = len(successfully_processed)
            estimated_cost = completed * 0.006
            monitor.log_progress(completed, estimated_cost)
            
            batch_info = {
                "batch_number": batch_num,
                "successfully_processed_in_session": completed - len(checkpoint_data[0] if checkpoint_data else []),
                "failed_in_session": len(batch_failed)
            }
            
            self.checkpoint_manager.save_checkpoint(
                successfully_processed, failed_indices, all_results, total_articles, batch_info
            )
            
            failure_rate = len(failed_indices) / (len(successfully_processed) + len(failed_indices)) if (successfully_processed or failed_indices) else 0
            if failure_rate > 0.1:
                delay = min(60, failure_rate * 300)
                logger.warning(f"High failure rate ({failure_rate:.1%}), adding {delay}s delay")
                await asyncio.sleep(delay)
        
        if failed_indices:
            logger.info(f"Final retry attempt for {len(failed_indices)} failed articles")
            final_retry_results, permanently_failed = await self.retry_failed_articles(failed_indices, articles)
            
            for result in final_retry_results:
                for idx, article in enumerate(articles):
                    if article["headline"] == result["headline"]:
                        successfully_processed.add(idx)
                        all_results.append(result)
                        failed_indices.discard(idx)
                        break
            
            failed_indices = permanently_failed
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_file = Path(self.config.output_dir) / f"perfect_ner_results_{timestamp}.json"
        
        with open(final_output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        if failed_indices:
            failure_report = {
                "permanently_failed_indices": list(failed_indices),
                "failed_articles": [
                    {"index": idx, "headline": articles[idx]["headline"]} 
                    for idx in failed_indices if idx < len(articles)
                ],
                "total_failed": len(failed_indices),
                "success_rate": len(successfully_processed) / total_articles
            }
            
            failure_file = Path(self.config.output_dir) / f"failed_articles_{timestamp}.json"
            with open(failure_file, "w", encoding="utf-8") as f:
                json.dump(failure_report, f, indent=2, ensure_ascii=False)
            
            logger.warning(f"Permanent failures saved to: {failure_file}")
        
        success_rate = len(successfully_processed) / total_articles
        if success_rate > 0.95:
            self.checkpoint_manager.clear_checkpoint()
            logger.info("Checkpoint cleared - high success rate achieved")
        else:
            logger.warning(f"Checkpoint retained - success rate only {success_rate:.1%}")
        
        total_entities = sum(r["entity_count"] for r in all_results)
        avg_entities = total_entities / len(all_results) if all_results else 0
        
        entity_type_counts = {}
        mode_counts = {}
        
        for result in all_results:
            mode_counts[result["mode"]] = mode_counts.get(result["mode"], 0) + 1
            for entity in result["entities"]:
                entity_type = entity["label"]
                entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        logger.info(f"\n=== ROBUST PROCESSING COMPLETE ===")
        logger.info(f"Successfully processed: {len(successfully_processed)}/{total_articles} ({success_rate:.1%})")
        logger.info(f"Permanent failures: {len(failed_indices)}")
        logger.info(f"Total entities extracted: {total_entities}")
        logger.info(f"Average entities per article: {avg_entities:.1f}")
        logger.info(f"Mode distribution: {mode_counts}")
        logger.info(f"Entity type breakdown: {entity_type_counts}")
        logger.info(f"Final results saved to: {final_output_file}")
        
        return all_results

    def load_articles(self) -> List[Dict]:
        """Load articles from data directory"""
        articles = []
        data_path = Path(self.pipeline.config["raw_data_dir"])
        
        if not data_path.exists():
            logger.warning(f"Data directory not found: {self.pipeline.config['raw_data_dir']}")
            return articles
        
        for json_file in data_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        articles.extend(data)
                    elif isinstance(data, dict):
                        articles.append(data)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
        
        articles = [a for a in articles if isinstance(a, dict) and a.get("headline")]
        logger.info(f"Loaded {len(articles)} articles")
        
        return articles

def load_articles(data_dir: str, n: int) -> List[Dict]:
    """Load random articles from JSON files (for original pipeline compatibility)"""
    articles = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return articles
    
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    articles.extend(data)
                elif isinstance(data, dict):
                    articles.append(data)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")
    
    articles = [a for a in articles if isinstance(a, dict) and a.get("headline")]
    
    return random.sample(articles, min(n, len(articles))) if articles else []

def estimate_processing_cost(num_articles: int = 15000):
    """Estimate API costs for processing"""
    avg_tokens_per_request = 800
    requests_needing_llm = num_articles * 1.2
    total_tokens = requests_needing_llm * avg_tokens_per_request
    
    input_cost = (total_tokens * 0.75) / 1_000_000 * 0.15
    output_cost = (total_tokens * 0.25) / 1_000_000 * 0.60
    
    total_cost = input_cost + output_cost
    processing_time_hours = requests_needing_llm / (350 * 60)
    
    print(f"=== COST ESTIMATION FOR {num_articles:,} ARTICLES ===")
    print(f"Estimated API requests: {requests_needing_llm:,.0f}")
    print(f"Estimated tokens: {total_tokens:,.0f}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"Estimated time: {processing_time_hours:.1f} hours")
    
    return total_cost, processing_time_hours

async def robust_main():
    """Main function with error handling and recovery"""
    
    caffeinate_process = setup_system_persistence()
    log_file = setup_logging_for_overnight()
    setup_macos_energy_settings()
    
    logger.info("Starting Complete Scalable Perfect NER Pipeline - macOS Overnight Mode")
    logger.info(f"Process ID: {os.getpid()}")
    logger.info(f"Log file: {log_file}")
    
    pipeline = PerfectNERPipeline(CONFIG)
    
    scaling_config = ScalingConfig(
        batch_size=20,
        max_concurrent_batches=2,
        rate_limit_requests_per_minute=350,
        max_articles=None,
        save_checkpoint_every=50,
        resume_from_checkpoint=True
    )
    
    total_cost, total_hours = estimate_processing_cost(15000)
    
    logger.info(f"Starting overnight processing:")
    logger.info(f"- Estimated cost: ${total_cost:.2f}")
    logger.info(f"- Estimated time: {total_hours:.1f} hours")
    logger.info(f"- Checkpoints every 50 articles")
    logger.info(f"- Conservative rate limits for stability")
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            processor = ScalablePipelineProcessor(pipeline, scaling_config)
            
            logger.info(f"Starting processing attempt {retry_count + 1}/{max_retries}")
            
            results = await processor.process_all_articles()
            
            logger.info("Processing completed successfully!")
            return results
            
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            if caffeinate_process:
                caffeinate_process.terminate()
            break
            
        except Exception as e:
            retry_count += 1
            wait_time = min(300, 60 * retry_count)
            
            logger.error(f"Processing failed (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                logger.error("Max retries exceeded. Check logs and resume manually.")
                if caffeinate_process:
                    caffeinate_process.terminate()
                raise
    
    if caffeinate_process:
        caffeinate_process.terminate()
    
    return []

def original_main():
    """Your original main function for testing/debugging"""
    logger.info("Starting Original Perfect NER Pipeline")
    
    pipeline = PerfectNERPipeline(CONFIG)
    articles = load_articles(CONFIG["raw_data_dir"], CONFIG["sample_size"])
    
    if not articles:
        logger.error("No articles found")
        return
    
    logger.info(f"Processing {len(articles)} articles")
    
    results = []
    mode_counts = {"ai_validated": 0, "llm_backup": 0, "forced_fallback": 0, "llm_validated_regex": 0}
    entity_type_counts = {}
    
    for i, article in enumerate(articles, 1):
        logger.info(f"Processing article {i}/{len(articles)}")
        
        text = f"{article['headline']} {article.get('summary', '')}"
        entities, mode = pipeline.extract_entities(text)
        
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        
        for entity in entities:
            entity_type = entity["label"]
            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        result = {
            "headline": article["headline"],
            "summary": article.get("summary", ""),
            "entities": entities,
            "entity_count": len(entities),
            "mode": mode
        }
        results.append(result)
        
        if entities:
            entity_summary = [f"{e['text']} ({e['label']})" for e in entities[:5]]
            logger.info(f"Found {len(entities)} entities: {', '.join(entity_summary)}")
        else:
            logger.info("No entities found")
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"perfect_ner_results_final_one.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_file}")
    
    total_entities = sum(r["entity_count"] for r in results)
    avg_entities = total_entities / len(results) if results else 0
    
    logger.info(f"\n=== ORIGINAL PIPELINE RESULTS ===")
    logger.info(f"Total articles: {len(results)}")
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Average entities per article: {avg_entities:.1f}")
    logger.info(f"Mode distribution: {mode_counts}")
    logger.info(f"Entity type breakdown: {entity_type_counts}")

async def main():
    """Main entry point - choose mode"""
    
    create_startup_script()
    
    
    results = await robust_main()
    
    
    logger.info("Pipeline execution completed")
    return results

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Check logs for details. You can resume by running the script again.")