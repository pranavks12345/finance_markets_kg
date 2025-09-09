#!/usr/bin/env python3
"""
Entity Analysis and Deduplication Pipeline
Extract all unique entities, analyze variations, and build mapping tables
"""

import json
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import pandas as pd
from pathlib import Path

def load_extraction_results(file_path: str) -> List[Dict]:
    """Load the extraction results"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_all_entities(results: List[Dict]) -> Dict[str, Set[str]]:
    """Extract all unique entities from relationships and events"""
    entities = {
        'companies': set(),
        'people': set(),
        'tickers': set(),
        'other': set()
    }
    
    for result in results:
        # From relationships
        for rel in result.get('relationships', []):
            source = rel.get('source_entity', '').strip()
            target = rel.get('target_entity', '').strip()
            
            if source:
                entities['companies'].add(source)
            if target:
                entities['companies'].add(target)
        
        # From events
        for event in result.get('events', []):
            primary_entity = event.get('primary_entity', '').strip()
            if primary_entity:
                entities['companies'].add(primary_entity)
    
    return entities

def classify_entities(all_entities: Set[str]) -> Dict[str, List[str]]:
    """Classify entities into companies, people, tickers, etc."""
    classified = {
        'companies': [],
        'people': [],
        'tickers': [],
        'government': [],
        'funds': [],
        'other': []
    }
    
    # Ticker patterns
    ticker_pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
    
    # Government/regulatory patterns
    government_patterns = [
        r'.*\b(SEC|FTC|DOJ|Fed|Federal Reserve|Treasury|Department|Commission|Agency|Government|Court)\b.*',
        r'.*\b(European Union|EU|Congress|Senate|House)\b.*',
        r'.*\b(Judge|Justice|Attorney General)\b.*'
    ]
    
    # Fund patterns
    fund_patterns = [
        r'.*\b(Fund|ETF|Index|Trust|Holdings|Partners|Capital|Ventures|Investment|Asset Management)\b.*',
        r'.*\b(Berkshire Hathaway|BlackRock|Vanguard|Fidelity)\b.*'
    ]
    
    # Person patterns (heuristic - names with 2+ words, first letters capitalized)
    person_pattern = r'^[A-Z][a-z]+ [A-Z][a-z]+( [A-Z][a-z]+)*$'
    
    for entity in all_entities:
        entity_clean = entity.strip()
        if not entity_clean:
            continue
            
        # Check if ticker
        if re.match(ticker_pattern, entity_clean):
            classified['tickers'].append(entity_clean)
        
        # Check if government/regulatory
        elif any(re.match(pattern, entity_clean, re.I) for pattern in government_patterns):
            classified['government'].append(entity_clean)
        
        # Check if fund
        elif any(re.match(pattern, entity_clean, re.I) for pattern in fund_patterns):
            classified['funds'].append(entity_clean)
        
        # Check if person (heuristic)
        elif re.match(person_pattern, entity_clean) and len(entity_clean.split()) >= 2:
            # Additional filters to avoid false positives
            if not any(word.lower() in ['inc', 'corp', 'llc', 'ltd', 'co', 'company', 'group', 'technologies', 'systems'] 
                      for word in entity_clean.split()):
                classified['people'].append(entity_clean)
            else:
                classified['companies'].append(entity_clean)
        
        # Default to company
        else:
            classified['companies'].append(entity_clean)
    
    return classified

def find_company_variations(companies: List[str]) -> Dict[str, List[str]]:
    """Find likely variations of the same company"""
    variations = defaultdict(list)
    
    # Common suffixes to normalize
    suffixes = ['Inc', 'Inc.', 'Corp', 'Corp.', 'LLC', 'Ltd', 'Ltd.', 'Co', 'Co.', 
                'Company', 'Corporation', 'Limited', 'Group', 'Holdings', 'Plc']
    
    def normalize_company_name(name: str) -> str:
        """Normalize company name for comparison"""
        normalized = name.strip()
        
        # Remove common suffixes
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)]
            elif normalized.endswith(' ' + suffix.lower()):
                normalized = normalized[:-len(' ' + suffix)]
        
        return normalized.lower().strip()
    
    # Group companies by normalized name
    company_groups = defaultdict(list)
    for company in companies:
        normalized = normalize_company_name(company)
        if normalized:  # Skip empty normalized names
            company_groups[normalized].append(company)
    
    # Only keep groups with multiple variations
    for normalized, group in company_groups.items():
        if len(group) > 1:
            # Sort by length (longer names first, assuming they're more complete)
            group_sorted = sorted(group, key=len, reverse=True)
            variations[normalized] = group_sorted
    
    return dict(variations)

def build_ticker_mapping() -> Dict[str, str]:
    """Build mapping of common tickers to company names"""
    # Common ticker mappings - you can expand this
    ticker_map = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft', 
        'GOOGL': 'Alphabet',
        'GOOG': 'Alphabet',
        'META': 'Meta',
        'TSLA': 'Tesla',
        'AMZN': 'Amazon',
        'NFLX': 'Netflix',
        'NVDA': 'Nvidia',
        'AMD': 'AMD',
        'INTC': 'Intel',
        'ORCL': 'Oracle',
        'IBM': 'IBM',
        'ADBE': 'Adobe',
        'CRM': 'Salesforce',
        'UBER': 'Uber',
        'SPOT': 'Spotify',
        'PYPL': 'PayPal',
        'SQ': 'Block',
        'TWTR': 'Twitter',
        'SNAP': 'Snap',
        'ZM': 'Zoom',
        'SHOP': 'Shopify',
        'CRWD': 'CrowdStrike',
        'PLTR': 'Palantir',
        'SNOW': 'Snowflake',
        'DBX': 'Dropbox',
        'DOCU': 'DocuSign',
        'OKTA': 'Okta',
        'TEAM': 'Atlassian',
        'MDB': 'MongoDB',
        'NET': 'Cloudflare',
        'DDOG': 'Datadog',
        'AVGO': 'Broadcom',
        'QCOM': 'Qualcomm',
        'MU': 'Micron',
        'TXN': 'Texas Instruments',
        'CSCO': 'Cisco',
        'V': 'Visa',
        'MA': 'Mastercard',
        'JPM': 'JPMorgan Chase',
        'BAC': 'Bank of America',
        'GS': 'Goldman Sachs',
        'MS': 'Morgan Stanley',
        'C': 'Citigroup',
        'WFC': 'Wells Fargo',
        'BRK.A': 'Berkshire Hathaway',
        'BRK.B': 'Berkshire Hathaway',
        'JNJ': 'Johnson & Johnson',
        'PFE': 'Pfizer',
        'UNH': 'UnitedHealth',
        'CVX': 'Chevron',
        'XOM': 'ExxonMobil',
        'WMT': 'Walmart',
        'KO': 'Coca-Cola',
        'DIS': 'Disney',
        'BA': 'Boeing',
        'CAT': 'Caterpillar',
        'GE': 'General Electric',
        'F': 'Ford',
        'GM': 'General Motors'
    }
    
    return ticker_map

def analyze_entity_frequency(results: List[Dict]) -> Dict[str, int]:
    """Count frequency of each entity across all relationships and events"""
    entity_counts = Counter()
    
    for result in results:
        # Count entities in relationships
        for rel in result.get('relationships', []):
            source = rel.get('source_entity', '').strip()
            target = rel.get('target_entity', '').strip()
            
            if source:
                entity_counts[source] += 1
            if target:
                entity_counts[target] += 1
        
        # Count entities in events
        for event in result.get('events', []):
            primary_entity = event.get('primary_entity', '').strip()
            if primary_entity:
                entity_counts[primary_entity] += 1
    
    return dict(entity_counts)

def save_analysis_results(classified_entities: Dict, variations: Dict, 
                         ticker_mapping: Dict, entity_frequencies: Dict, 
                         output_dir: str = "entity_analysis"):
    """Save all analysis results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save classified entities
    with open(output_path / "classified_entities.json", "w", encoding="utf-8") as f:
        json.dump(classified_entities, f, indent=2, ensure_ascii=False)
    
    # Save company variations
    with open(output_path / "company_variations.json", "w", encoding="utf-8") as f:
        json.dump(variations, f, indent=2, ensure_ascii=False)
    
    # Save ticker mapping
    with open(output_path / "ticker_mapping.json", "w", encoding="utf-8") as f:
        json.dump(ticker_mapping, f, indent=2, ensure_ascii=False)
    
    # Save entity frequencies
    with open(output_path / "entity_frequencies.json", "w", encoding="utf-8") as f:
        json.dump(entity_frequencies, f, indent=2, ensure_ascii=False)
    
    # Create summary CSV files for easy review
    
    # Companies with frequencies
    companies_df = pd.DataFrame([
        {"entity": entity, "frequency": entity_frequencies.get(entity, 0), "type": "company"}
        for entity in classified_entities['companies']
    ]).sort_values('frequency', ascending=False)
    companies_df.to_csv(output_path / "companies_ranked.csv", index=False)
    
    # People with frequencies  
    people_df = pd.DataFrame([
        {"entity": entity, "frequency": entity_frequencies.get(entity, 0), "type": "person"}
        for entity in classified_entities['people']
    ]).sort_values('frequency', ascending=False)
    people_df.to_csv(output_path / "people_ranked.csv", index=False)
    
    # Tickers with frequencies
    tickers_df = pd.DataFrame([
        {"entity": entity, "frequency": entity_frequencies.get(entity, 0), "type": "ticker"}
        for entity in classified_entities['tickers']
    ]).sort_values('frequency', ascending=False)
    tickers_df.to_csv(output_path / "tickers_ranked.csv", index=False)
    
    print(f"Analysis results saved to {output_path}/")

def print_analysis_summary(classified_entities: Dict, variations: Dict, 
                         entity_frequencies: Dict):
    """Print summary of entity analysis"""
    print("\n" + "="*60)
    print("ENTITY ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total unique companies: {len(classified_entities['companies'])}")
    print(f"Total unique people: {len(classified_entities['people'])}")  
    print(f"Total unique tickers: {len(classified_entities['tickers'])}")
    print(f"Government/regulatory entities: {len(classified_entities['government'])}")
    print(f"Funds/investment entities: {len(classified_entities['funds'])}")
    print(f"Other entities: {len(classified_entities['other'])}")
    
    print(f"\nCompany variations found: {len(variations)} groups")
    
    # Show top entities by frequency
    sorted_frequencies = sorted(entity_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 most mentioned entities:")
    for i, (entity, count) in enumerate(sorted_frequencies[:20], 1):
        print(f"{i:2d}. {entity}: {count} mentions")
    
    print(f"\nCompany variations examples:")
    for i, (base_name, variants) in enumerate(list(variations.items())[:5], 1):
        print(f"{i}. {base_name}: {variants}")
    
    print("="*60)

def main():
    """Main analysis pipeline"""
    
    # Configuration
    INPUT_FILE = "extracted_relationships_events_FULL.json"
    
    print("Loading extraction results...")
    results = load_extraction_results(INPUT_FILE)
    print(f"Loaded {len(results)} articles")
    
    print("\nExtracting all entities...")
    all_entities_raw = extract_all_entities(results)
    all_unique_entities = all_entities_raw['companies']  # All entities were classified as companies initially
    print(f"Found {len(all_unique_entities)} unique entities")
    
    print("\nClassifying entities...")
    classified_entities = classify_entities(all_unique_entities)
    
    print("\nFinding company variations...")
    company_variations = find_company_variations(classified_entities['companies'])
    
    print("\nBuilding ticker mapping...")
    ticker_mapping = build_ticker_mapping()
    
    print("\nAnalyzing entity frequencies...")
    entity_frequencies = analyze_entity_frequency(results)
    
    print("\nSaving analysis results...")
    save_analysis_results(classified_entities, company_variations, 
                         ticker_mapping, entity_frequencies)
    
    print_analysis_summary(classified_entities, company_variations, entity_frequencies)

if __name__ == "__main__":
    main()