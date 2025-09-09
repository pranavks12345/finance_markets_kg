

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
import hashlib
import asyncio
import json
import re
import logging
import aiohttp
import ssl

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ImportError as e:
    print(f"Missing required packages. Please install: pip install openai python-dotenv")
    exit(1)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionGoldGenerator:
    def __init__(self):
        self.MAX_ARTICLES = 5000
        self.COST_PER_ARTICLE = 0.10
        self.BATCH_SIZE = 200
        
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        
        self.company_normalizer = self._build_company_normalizer()
        
       
        self.entity_stats = Counter()
        self.quality_issues = []
        self.duplicate_tracker = defaultdict(set)
        
        logger.info("Production Gold Generator initialized")

    def _build_company_normalizer(self) -> Dict[str, str]:
        """Comprehensive canonical company name mappings"""
        return {
            
            "apple": "Apple Inc.",
            "apple inc": "Apple Inc.", 
            "apple inc.": "Apple Inc.",
            "microsoft": "Microsoft Corporation",
            "microsoft corp": "Microsoft Corporation",
            "microsoft corporation": "Microsoft Corporation",
            "alphabet": "Alphabet Inc.",
            "alphabet inc": "Alphabet Inc.",
            "google": "Alphabet Inc.",
            "tesla": "Tesla Inc.",
            "amazon": "Amazon.com Inc.",
            "nvidia": "NVIDIA Corporation",
            "meta": "Meta Platforms Inc.",
            "meta platforms": "Meta Platforms Inc.",
            "facebook": "Meta Platforms Inc.",
            "qualcomm": "Qualcomm Inc.",
            "garmin": "Garmin Ltd.",
            
            
            "berkshire hathaway": "Berkshire Hathaway Inc.",
            "berkshire": "Berkshire Hathaway Inc.",
            "jpmorgan": "JPMorgan Chase & Co.",
            "jpmorgan chase": "JPMorgan Chase & Co.",
            "jp morgan": "JPMorgan Chase & Co.",
            "goldman sachs": "Goldman Sachs Group Inc.",
            "morgan stanley": "Morgan Stanley",
            "bank of america": "Bank of America Corporation",
            "wells fargo": "Wells Fargo & Company",
            "citigroup": "Citigroup Inc.",
            "citi": "Citigroup Inc.",
            "ubs": "UBS Group AG",
            "jefferies": "Jefferies Financial Group Inc.",
            
            "walmart": "Walmart Inc.",
            "exxonmobil": "Exxon Mobil Corporation",
            "exxon mobil": "Exxon Mobil Corporation",
            "johnson & johnson": "Johnson & Johnson",
            "procter & gamble": "Procter & Gamble Company",
            "coca-cola": "Coca-Cola Company",
            "visa": "Visa Inc.",
            "mastercard": "Mastercard Incorporated",
            "american express": "American Express Company",
            "servicenow": "ServiceNow Inc.",
            "robinhood": "Robinhood Markets Inc.",
            "ford": "Ford Motor Company",
            
            
            "disney": "Walt Disney Company",
            "comcast": "Comcast Corporation",
            "netflix": "Netflix Inc.",
            "graniteshares": "GraniteShares Ltd.",
            
            
            "franklin rising dividends sma": "Franklin Templeton",
            "telness tech": "Telness Tech AB",
            "smallstep": "Smallstep Inc.",
            "jamf": "Jamf Holding Corp.",
        }

    def _build_enhanced_prompt(self) -> str:
        """Enhanced prompt optimized for maximum recall"""
        return """
TASK: Extract EVERY financial entity from this news article. Your goal is maximum coverage - don't miss anything.

ARTICLE TEXT:
"{headline} {summary}"

EXTRACTION STRATEGY:
1. Read through the text multiple times
2. Extract ALL companies, people, money amounts, percentages, tickers, dates
3. Include obvious entities even if they seem generic
4. When in doubt, include it - false positives are better than missed entities

CATEGORIES (extract liberally):

COMPANY: Any business name mentioned
- Apple, Microsoft, Tesla, Meta, Amazon, Nvidia, Alphabet
- Goldman Sachs, Morgan Stanley, UBS, JPMorgan
- Qualcomm, Ford, ServiceNow, Robinhood, Garmin
- Extract as written: "Apple" not "Apple Inc."


PERSON: Any individual mentioned
- Warren Buffett, Donald Trump, Tim Cook, Elon Musk
- Include titles: "President Trump", "CEO Cook"
- First name + last name combinations

TICKER: Stock symbols
- AAPL, MSFT, TSLA, META, AMZN, NVDA, GOOGL
- Include exchange: NASDAQ:AAPL, NYSE:BRK.B

MONEY: All monetary amounts
- $1.5 billion, €50M, USD 240, $210
- Don't miss: $53 million, $3.45 trillion

PERCENTAGE: All percentages
- 15%, 0.7%, 25 basis points, 6% growth
- Even small ones: 2.9%, -13.61%

TIME_PERIOD: All time references
- Q2, Q3, 2025, July 22, fiscal year
- "second quarter", "March 31, 2025"

PRODUCT: Named products/services
- iPhone, iPhone 18, ChatGPT, Model S
- "GPS-enabled fitness devices"

INDEX: Market indices
- S&P 500, Dow Jones, Nasdaq, Russell 2000
- "NYSE Health Care Index"

LOCATION: Places mentioned
- San Francisco, New York, Stockholm
- Wall Street, Silicon Valley

GOV_ORG: Government entities
- Federal Reserve, SEC, Trump administration
- "Commerce Department"

PUBLISHER: Media companies
- Bloomberg, Reuters, Wall Street Journal
- Jefferies (when used as source)

CONCEPT: Financial/business terms but dont overdo it and keep it related not too generic
- artificial intelligence, digital transformation
- "free cash flow", "supply chain"

POLICY: Government policies but again do not be too generic be specific and a little strict
- tariffs, CHIPS Act, regulations

COUNTRY: Nations mentioned
- United States, China, US, EU

EVENT: Named events but dont overdo it and be specific and strict on whats an event
- earnings call, FOMC meeting

ALIAS: Group nicknames
- "Magnificent Seven", "Big Tech"

CRITICAL REMINDERS:
- Extract "Apple" not "Apple Inc." - normalization happens later
- Don't skip obvious tickers like AAPL, MSFT
- Include ALL money amounts: $1B, $240, €50M
- Include ALL percentages: 15%, 0.5%, 10 bps
- Include ALL time periods: Q2, 2025, July 30
- When unsure, INCLUDE IT

OUTPUT (JSON array only):
[
  {{"text": "entity as written in text", "label": "CATEGORY"}}
]"""

    def post_process_cleanup(self, entities: List[Dict], text: str) -> List[Dict]:
        """Comprehensive post-processing cleanup to fix common issues"""
        cleaned_entities = []
        text_lower = text.lower()
        
        for entity in entities:
            original_text = entity["text"]
            label = entity["label"].upper()
            
            
            if original_text.lower() not in text_lower:
                continue
            
            cleaned_text = original_text.strip()
            
            
            if label == "COMPANY":
                
                if text.lower() in ["big tech", "magnificent seven", "mag 7"]:
                    label = "CONCEPT"  
                else:
                    cleaned_text = self._cleanup_company_name(cleaned_text)
            elif label == "PERSON":
                cleaned_text = self._cleanup_person_name(cleaned_text, text)
            elif label == "TICKER":
                cleaned_text = self._cleanup_ticker(cleaned_text)
            elif label == "MONEY":
                if not self._is_valid_money(cleaned_text):
                    continue
                cleaned_text = self._cleanup_money(cleaned_text)
            elif label == "PERCENTAGE":
                if not self._is_valid_percentage(cleaned_text):
                    continue
            elif label == "INDEX":
                cleaned_text = self._cleanup_index_name(cleaned_text)
            elif label == "PUBLISHER":
                cleaned_text = self._cleanup_publisher_name(cleaned_text)
            elif label == "CONCEPT":
                if cleaned_text.lower() == "ai":
                    cleaned_text = "artificial intelligence"
            
            cleaned_text = self._generic_text_cleanup(cleaned_text)
            
            if self._final_entity_validation(cleaned_text, label, text):
                cleaned_entities.append({
                    "text": cleaned_text,
                    "label": label,
                    "original_text": original_text
                })
        
        cleaned_entities = self._split_compound_entities(cleaned_entities, text)
        
        try:
            missed_entities = self._identify_missed_entities(text, cleaned_entities)
            cleaned_entities.extend(missed_entities)
        except AttributeError:
            pass
        
        cleaned_entities = self._remove_duplicates(cleaned_entities)
        
        return cleaned_entities

    def _identify_missed_entities(self, text: str, existing_entities: List[Dict]) -> List[Dict]:
        """Identify obvious entities that the LLM missed"""
        missed_entities = []
        existing_texts = {e["text"].lower() for e in existing_entities}
        
        money_patterns = [
            r'\$[\d,.]+ ?(?:billion|million|thousand|B|M|K)\b',
            r'\$[\d,.]+\b',
            r'USD [\d,.]+\b',
            r'€[\d,.]+ ?(?:billion|million|B|M)?\b',
            r'£[\d,.]+ ?(?:billion|million|B|M)?\b'
        ]
        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.I):
                entity_text = match.group().strip()
                if (entity_text.lower() not in existing_texts and 
                    self._is_valid_money(entity_text)):
                    missed_entities.append({
                        "text": entity_text,
                        "label": "MONEY",
                        "original_text": entity_text
                    })
        
        pct_patterns = [
            r'\d+(?:\.\d+)?%',
            r'\d+(?:\.\d+)? ?(?:percent|basis points?|bps)\b'
        ]
        for pattern in pct_patterns:
            for match in re.finditer(pattern, text, re.I):
                entity_text = match.group().strip()
                if entity_text.lower() not in existing_texts:
                    missed_entities.append({
                        "text": entity_text,
                        "label": "PERCENTAGE",
                        "original_text": entity_text
                    })
        
        time_patterns = [
            r'\bQ[1-4](?:\s+20\d{2})?\b',
            r'\b(?:first|second|third|fourth) quarter\b',
            r'\b(?:fiscal )?20\d{2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b'
        ]
        for pattern in time_patterns:
            for match in re.finditer(pattern, text, re.I):
                entity_text = match.group().strip()
                if entity_text.lower() not in existing_texts:
                    missed_entities.append({
                        "text": entity_text,
                        "label": "TIME_PERIOD",
                        "original_text": entity_text
                    })
        
        major_companies = [
            'Apple', 'Microsoft', 'Tesla', 'Amazon', 'Google', 'Meta', 'Nvidia', 'Alphabet',
            'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'UBS', 'Jefferies', 
            'Qualcomm', 'Ford', 'Garmin', 'ServiceNow', 'Robinhood'
        ]
        
        for company in major_companies:
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, text) and company.lower() not in existing_texts:
                missed_entities.append({
                    "text": company,
                    "label": "COMPANY",
                    "original_text": company
                })
        
        ticker_candidates = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA', 'QCOM']
        ticker_pattern = r'\b(' + '|'.join(ticker_candidates) + r')\b'
        for match in re.finditer(ticker_pattern, text):
            entity_text = match.group().strip()
            if entity_text.lower() not in existing_texts:
                missed_entities.append({
                    "text": entity_text,
                    "label": "TICKER",
                    "original_text": entity_text
                })
        
        person_patterns = [
            r'\b(?:Donald Trump|Warren Buffett|Tim Cook|Elon Musk|Jerome Powell)\b',
            r'\b(?:President (?:Trump|Biden|Obama))\b',
            r'\b(?:CEO|Chairman|Founder) [A-Z][a-z]+ [A-Z][a-z]+\b'
        ]
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                entity_text = match.group().strip()
                if entity_text.lower() not in existing_texts:
                    missed_entities.append({
                        "text": entity_text,
                        "label": "PERSON",
                        "original_text": entity_text
                    })
        
        index_patterns = [
            r'\bS&P 500\b',
            r'\bDow Jones\b',
            r'\bNasdaq\b',
            r'\bRussell 2000\b'
        ]
        for pattern in index_patterns:
            for match in re.finditer(pattern, text, re.I):
                entity_text = match.group().strip()
                if entity_text.lower() not in existing_texts:
                    missed_entities.append({
                        "text": entity_text,
                        "label": "INDEX",
                        "original_text": entity_text
                    })
        
        return missed_entities
    def safe_parse_entities(self, content: str):
        """Robust JSON parsing for LLM output"""
        if not content:
            return []

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content, flags=re.I).strip()
            if content.endswith("```"):
                content = content[:-3].strip()

        start, end = content.find("["), content.rfind("]")
        if start != -1 and end != -1:
            content = content[start:end+1]

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
            else:
                return []
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e} | raw: {content[:120]}")
            return []


    def _extract_basic_entities(self, text: str) -> List[Dict]:
        """Fallback extraction for when LLM returns zero entities or very few"""
        basic_entities = []
        
        money_patterns = [
            r'\$[\d,.]+ ?(?:billion|million|thousand|B|M|K)\b',
            r'\$[\d,.]+\b',
            r'USD [\d,.]+\b',
            r'€[\d,.]+ ?(?:billion|million|B|M)?\b',
            r'£[\d,.]+ ?(?:billion|million|B|M)?\b'
        ]
        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.I):
                entity_text = match.group().strip()
                if self._is_valid_money(entity_text):
                    basic_entities.append({
                        "text": entity_text,
                        "label": "MONEY",
                        "original_text": entity_text
                    })
        
        pct_patterns = [r'\d+(?:\.\d+)?%', r'\d+(?:\.\d+)? ?(?:percent|basis points?|bps)\b']
        for pattern in pct_patterns:
            for match in re.finditer(pattern, text, re.I):
                basic_entities.append({
                    "text": match.group().strip(),
                    "label": "PERCENTAGE",
                    "original_text": match.group().strip()
                })
        
        obvious_companies = [
            'Apple', 'Microsoft', 'Tesla', 'Amazon', 'Google', 'Meta', 'Nvidia', 'Alphabet',
            'Goldman Sachs', 'Morgan Stanley', 'JPMorgan', 'UBS', 'Qualcomm', 'Ford',
            'Garmin', 'ServiceNow', 'Robinhood', 'Jefferies'
        ]
        for company in obvious_companies:
            pattern = r'\b' + re.escape(company) + r'\b'
            if re.search(pattern, text):
                basic_entities.append({
                    "text": company,
                    "label": "COMPANY",
                    "original_text": company
                })
        
        ticker_candidates = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'GOOG', 'META', 'AMZN', 'NVDA']
        for ticker in ticker_candidates:
            if re.search(r'\b' + ticker + r'\b', text):
                basic_entities.append({
                    "text": ticker,
                    "label": "TICKER",
                    "original_text": ticker
                })
        
        time_patterns = [
            r'\bQ[1-4](?:\s+20\d{2})?\b',
            r'\b20\d{2}\b',
            r'\b(?:first|second|third|fourth) quarter\b',
            r'\bfiscal year\b'
        ]
        for pattern in time_patterns:
            for match in re.finditer(pattern, text, re.I):
                basic_entities.append({
                    "text": match.group().strip(),
                    "label": "TIME_PERIOD",
                    "original_text": match.group().strip()
                })
        
        person_patterns = [
            r'\bWarren Buffett\b',
            r'\bDonald Trump\b', 
            r'\bTim Cook\b',
            r'\bElon Musk\b',
            r'\bJerome Powell\b'
        ]
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                basic_entities.append({
                    "text": match.group().strip(),
                    "label": "PERSON",
                    "original_text": match.group().strip()
                })
        
        index_patterns = [
            r'\bS&P 500\b',
            r'\bDow Jones\b',
            r'\bNasdaq\b'
        ]
        for pattern in index_patterns:
            for match in re.finditer(pattern, text):
                basic_entities.append({
                    "text": match.group().strip(),
                    "label": "INDEX",
                    "original_text": match.group().strip()
                })
        
        return basic_entities

    def _cleanup_company_name(self, text: str) -> str:
        """Clean and normalize company names"""
        text_lower = text.lower().strip()
        
        if text_lower in self.company_normalizer:
            return self.company_normalizer[text_lower]
        
        if text_lower.endswith(" corp"):
            base_name = text_lower[:-5]
            if base_name in self.company_normalizer:
                return self.company_normalizer[base_name]
        
        if text_lower.endswith(" inc"):
            base_name = text_lower[:-4]
            if base_name in self.company_normalizer:
                return self.company_normalizer[base_name]
        
        return text.strip()

    def _cleanup_person_name(self, text: str, context: str) -> str:
        """Clean and validate person names"""
        text = text.strip()
        
        text = re.sub(r'^(the\s+|mr\.?\s+|ms\.?\s+|mrs\.?\s+|dr\.?\s+)', '', text, flags=re.I)
        text = re.sub(r'\s+(said|says|stated|announced)$', '', text, flags=re.I)
        
        words = text.split()
        if len(words) >= 2:
            text = ' '.join(word.capitalize() if word.isalpha() else word for word in words)
        
        return text

    def _cleanup_ticker(self, text: str) -> str:
        """Clean and standardize ticker symbols"""
        text = text.strip().upper()
        
        if text.startswith('(') and text.endswith(')'):
            text = text[1:-1]
        
        return text

    def _is_valid_money(self, text: str) -> bool:
        """Validate money entities"""
        has_digit = re.search(r'\d', text)
        has_currency = re.search(r'[\$€£¥]|USD|EUR|GBP|JPY|billion|million|thousand|B|M|K', text, re.I)
        
        if re.match(r'^\d+$', text):
            return False
        
        return has_digit and has_currency

    def _cleanup_money(self, text: str) -> str:
        """Clean money amounts"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\$\s+', '$', text)
        text = re.sub(r'(\d)\s*([BMK])\b', r'\1 \2', text, flags=re.I)
        return text.strip()

    def _is_valid_percentage(self, text: str) -> bool:
        """Validate percentage entities"""
        return bool(re.search(r'\d+.*(?:%|percent|basis|bps)', text, re.I))

    def _cleanup_index_name(self, text: str) -> str:
        """Clean and standardize index names"""
        text = text.strip()
        
        expansions = {
            "s&p": "S&P 500",
            "dow": "Dow Jones Industrial Average", 
            "nasdaq": "Nasdaq Composite",
            "russell 2000": "Russell 2000 Index"
        }
        
        text_lower = text.lower()
        return expansions.get(text_lower, text)

    def _cleanup_publisher_name(self, text: str) -> str:
        """Clean publisher names"""
        corrections = {
            "wsj": "Wall Street Journal",
            "ft": "Financial Times",
            "nyt": "New York Times",
            "wapo": "Washington Post"
        }
        
        text_lower = text.lower().strip()
        return corrections.get(text_lower, text.strip())

    def _generic_text_cleanup(self, text: str) -> str:
        """Apply generic text cleanup"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[,;]+$', '', text)
        text = re.sub(r'\s+(reportedly|said|says|stated|announced|according)$', '', text, flags=re.I)
        return text

    def _final_entity_validation(self, text: str, label: str, context: str) -> bool:
        """Final validation before accepting entity"""
        if len(text.strip()) < 1:
            return False
        
        noise_words = {
            'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those',
            'company', 'firm', 'market', 'sector', 'industry', 'stock',
            'the', 'and', 'or', 'but', 'for', 'on', 'in', 'at', 'to'
        }
        
        if text.lower().strip() in noise_words:
            return False
        
        if label == "PERSON" and len(text.split()) < 2:
            person_indicators = ['ceo', 'president', 'chairman', 'founder', 'analyst']
            if not any(indicator in context.lower() for indicator in person_indicators):
                return False
        
        if label == "TICKER" and not re.match(r'^[A-Z]+:[A-Z]{1,5}$|^[A-Z]{1,5}$', text):
            return False
        
        return True

    def _split_compound_entities(self, entities: List[Dict], text: str) -> List[Dict]:
        """Split compound entities like 'Apple (NASDAQ:AAPL)' into separate entities"""
        new_entities = []
        
        for entity in entities:
            original_text = entity["text"]
            
            match = re.match(r'^(.+?)\s*\(([A-Z]+:[A-Z]{1,5}|[A-Z]{1,5})\)$', original_text)
            if match and entity["label"] == "COMPANY":
                company_name = match.group(1).strip()
                ticker = match.group(2).strip()
                
                new_entities.append({
                    "text": self._cleanup_company_name(company_name),
                    "label": "COMPANY",
                    "original_text": original_text
                })
                
                new_entities.append({
                    "text": ticker,
                    "label": "TICKER", 
                    "original_text": original_text
                })
            else:
                new_entities.append(entity)
        
        return new_entities

    def _remove_duplicates(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities while preserving order"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity["text"].lower(), entity["label"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities

    

  

   
    async def fetch_entities(self, session, headline, summary, semaphore):
        """True async call to OpenAI API using aiohttp with retry logic"""
        prompt = self._build_enhanced_prompt().format(headline=headline, summary=summary)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a financial NER expert. Extract ALL entities."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 1000
        }
        async with semaphore:
            for attempt in range(3):
                try:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status == 429:
                            retry_after = resp.headers.get('retry-after-ms', 1000)
                            wait_time = int(retry_after) / 1000.0 if retry_after else 1.0
                            print(f"[INFO] Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if resp.status != 200:
                            error_text = await resp.text()
                            print(f"[WARN] API HTTP {resp.status}: {error_text[:200]}")
                            return ""
                        
                        data = await resp.json()
                        
                        if 'error' in data:
                            print(f"[WARN] OpenAI API error: {data['error']}")
                            return ""
                        
                        if 'choices' not in data or len(data['choices']) == 0:
                            print(f"[WARN] Unexpected API response structure")
                            return ""
                        
                        return data["choices"][0]["message"]["content"]
                        
                except Exception as e:
                    print(f"[WARN] Request exception (attempt {attempt+1}): {e}")
                    if attempt == 2:
                        return ""
                    await asyncio.sleep(1)
        
        return ""

    async def process_batch_async(self, batch_articles, batch_num):
        """Process a batch of articles in parallel"""
        semaphore = asyncio.Semaphore(5)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for article in batch_articles:
                headline = article.get("headline", "").strip()
                summary = article.get("summary", "").strip()
                if (headline and len(headline) >= 5) or (summary and len(summary) >= 10):
                    tasks.append(self.fetch_entities(session, headline, summary, semaphore))

            if not tasks:
                print(f"[WARN] No valid articles in batch {batch_num}")
                return []

            results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        task_index = 0
        for article in batch_articles:
            headline = article.get("headline", "").strip()
            summary = article.get("summary", "").strip()
            
            if not ((headline and len(headline) >= 5) or (summary and len(summary) >= 10)):
                continue
                
            try:
                content = results[task_index]
                task_index += 1
                
                if isinstance(content, Exception):
                    print(f"[WARN] API error: {content}")
                    continue

                content = content.strip()

                if content.startswith("```"):
                    content = re.sub(r"^```(json)?", "", content, flags=re.IGNORECASE).strip()
                    content = content.rstrip("`").strip()

                try:
                    entities = self.safe_parse_entities(content)
                    if not isinstance(entities, list):
                        continue
                except json.JSONDecodeError as e:
                    print(f"[WARN] Bad JSON in article: {e} -> {content[:200]}")
                    continue

                full_text = f"{headline} {summary}"
                cleaned_entities = self.post_process_cleanup(entities, full_text)
                processed.append({
                    "headline": headline,
                    "summary": summary,
                    "entities": cleaned_entities
                })
            except Exception as e:
                print(f"[WARN] Failed parsing article: {e}")
        return processed   




    def generate_dataset(self) -> List[Dict]:
        """Generate production dataset with quality controls"""
        data_dir = Path("data/raw/news")
        if not data_dir.exists():
            logger.error("No data directory found")
            return []
        
        articles = []
        for json_file in data_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    articles.extend(data)
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        if not articles:
            logger.error("No articles found")
            return []
        
        num_articles = min(len(articles), self.MAX_ARTICLES)
        estimated_cost = num_articles * self.COST_PER_ARTICLE
        
        logger.info(f"Processing {num_articles} articles")
        logger.info(f"Estimated cost: ${estimated_cost:.2f}")
        
        all_results = []
        
        for batch_start in range(0, num_articles, self.BATCH_SIZE):
            batch_num = (batch_start // self.BATCH_SIZE) + 1
            batch_end = min(batch_start + self.BATCH_SIZE, num_articles)
            batch_articles = articles[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_num}: articles {batch_start+1}-{batch_end}")
            
            batch_results = asyncio.run(self.process_batch_async(batch_articles, batch_num))

            all_results.extend(batch_results)
            
            total_entities = sum(len(r["entities"]) for r in all_results)
            priority_count = sum(1 for r in all_results if r.get("priority", False))
            
            logger.info(f"Batch {batch_num} complete: {len(batch_results)} articles, {total_entities} total entities")
            logger.info(f"Priority articles so far: {priority_count}")
            
            if batch_num % 10 == 0:
                top_entities = self.entity_stats.most_common(8)
                logger.info(f"Top entity types: {top_entities}")
        
        return all_results

    def save_results(self, results: List[Dict]) -> None:
        """Save results with comprehensive quality report"""
        output_file = Path("production_gold_dataset_5k.json")
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self._generate_quality_report(results, "production_quality_report.json")
        
        total_entities = sum(len(r["entities"]) for r in results)
        priority_articles = sum(1 for r in results if r.get("priority", False))
        
        logger.info(f"\n{'='*50}")
        logger.info(f"PRODUCTION DATASET COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Articles processed: {len(results)}")
        logger.info(f"Total entities: {total_entities}")
        logger.info(f"Average entities per article: {total_entities/len(results):.1f}")
        logger.info(f"Priority articles: {priority_articles}")
        logger.info(f"Quality issues: {len(self.quality_issues)}")
        logger.info(f"Saved to: {output_file}")

    def _generate_quality_report(self, results: List[Dict], report_file: str) -> None:
        """Generate comprehensive quality report"""
        total_entities = sum(len(r["entities"]) for r in results)
        
        entity_percentages = {
            label: (count / total_entities * 100) 
            for label, count in self.entity_stats.items()
        }
        
        entities_by_type = defaultdict(Counter)
        for result in results:
            for entity in result["entities"]:
                entities_by_type[entity["label"]][entity["text"]] += 1
        
        max_count = max(self.entity_stats.values()) if self.entity_stats else 0
        min_count = min(self.entity_stats.values()) if self.entity_stats else 0
        balance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        report = {
            "dataset_stats": {
                "total_articles": len(results),
                "total_entities": total_entities,
                "avg_entities_per_article": total_entities / len(results) if results else 0,
                "priority_articles": sum(1 for r in results if r.get("priority", False)),
                "quality_issues": len(self.quality_issues),
                "balance_ratio": balance_ratio
            },
            "entity_distribution": dict(self.entity_stats),
            "entity_percentages": entity_percentages,
            "top_entities_by_type": {
                label: counter.most_common(10) 
                for label, counter in entities_by_type.items()
            },
            "quality_issues": self.quality_issues[:100],
            "recommendations": self._generate_recommendations(balance_ratio, entity_percentages)
        }
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {report_file}")

    def _generate_recommendations(self, balance_ratio: float, entity_percentages: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if balance_ratio > 20:
            recommendations.append("Dataset is highly imbalanced - consider targeted sampling for rare entity types")
        
        priority_categories = {"POLICY", "GOV_ORG", "ALIAS", "INDEX", "EVENT"}
        for category in priority_categories:
            if entity_percentages.get(category, 0) < 3:
                recommendations.append(f"Category {category} is underrepresented - find more relevant articles")
        
        if len(self.quality_issues) > len(self.entity_stats) * 50:
            recommendations.append("High error rate detected - review prompt clarity and validation rules")
        
        return recommendations


def main():
    """Main execution"""
    generator = ProductionGoldGenerator()
    results = generator.generate_dataset()
    
    if results:
        generator.save_results(results)
        logger.info("Production gold dataset generation complete!")
        logger.info("Ready for BERT fine-tuning and model training")
    else:
        logger.error("Dataset generation failed")


if __name__ == "__main__":
    main()