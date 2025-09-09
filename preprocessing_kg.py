#!/usr/bin/env python3
"""
Entity Deduplication and Mapping Pipeline
Clean up entity variations and create canonical mappings for knowledge graph
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import pandas as pd
from pathlib import Path

class EntityDeduplicator:
    def __init__(self):
        self.canonical_mappings = {}
        self.ticker_to_company = self._build_ticker_mapping()
        self.blacklist_entities = self._build_blacklist()
        
    def _build_ticker_mapping(self) -> Dict[str, str]:
        """Enhanced ticker to company mapping"""
        return {
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
            'SNAP': 'Snap',
            'ZM': 'Zoom',
            'SHOP': 'Shopify',
            'CRWD': 'CrowdStrike',
            'PLTR': 'Palantir',
            'SNOW': 'Snowflake',
            'TEAM': 'Atlassian',
            'NET': 'Cloudflare',
            'AVGO': 'Broadcom',
            'QCOM': 'Qualcomm',
            'MU': 'Micron',
            'CSCO': 'Cisco',
            'V': 'Visa',
            'MA': 'Mastercard',
            'JPM': 'JPMorgan Chase',
            'BAC': 'Bank of America',
            'GS': 'Goldman Sachs',
            'MS': 'Morgan Stanley',
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
            'GM': 'General Motors',
            'TSMC': 'Taiwan Semiconductor',
            'TSM': 'Taiwan Semiconductor'
        }
    
    def _build_blacklist(self) -> Set[str]:
        """Entities that should be excluded from the knowledge graph"""
        return {
            # Generic terms
            'market', 'markets', 'earnings', 'competitors', 'analysts', 'investors',
            'companies', 'shareholders', 'employees', 'peers', 'rivals', 'estimates',
            'expectations', 'buy', 'benchmark', 'dollar', 'gold', 'copper', 'bitcoin',
            
            # Abstract concepts  
            'ai', 'ai stocks', 'ai stock', 'ai trade', 'ai technology', 'technology stocks',
            'us stocks', 'us equity markets', 'us stock indexes', 'quantum computing stocks',
            'semiconductor industry', 'robotics', 'logistics', 'e-commerce',
            
            # Market indices/concepts
            'nasdaq composite', 'most valuable', "world's most valuable", 'hyperscale cloud computing companies',
            'nlp in healthcare & life sciences market', 'ibd 50',
            
            # Job references
            '9,000 jobs', 'tariffs',
            
            # Generic descriptors
            'u.s.', 'u.s. companies', 'unknown', 'unknown company', 'robotaxi'
        }
    
    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for deduplication"""
        if not name:
            return ""
            
        # Convert to lowercase for processing
        normalized = name.lower().strip()
        
        # Skip if in blacklist
        if normalized in self.blacklist_entities:
            return ""
        
        # Remove common suffixes
        suffixes = ['inc.', 'inc', 'corp.', 'corp', 'company', 'co.', 'co', 'ltd.', 'ltd', 
                   'llc', 'group', 'holdings', 'plc', 'corporation', 'limited']
        
        for suffix in suffixes:
            if normalized.endswith(' ' + suffix):
                normalized = normalized[:-len(' ' + suffix)].strip()
        
        # Handle special cases
        normalized = normalized.replace('&', 'and')
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        return normalized
    
    def build_canonical_mappings(self, company_variations: Dict[str, List[str]]) -> Dict[str, str]:
        """Build canonical entity mappings"""
        mappings = {}
        
        # Process company variations from analysis
        for base_name, variants in company_variations.items():
            if not variants:
                continue
                
            # Choose canonical name (longest variant, assuming it's most complete)
            canonical = max(variants, key=len)
            
            # Clean up canonical name
            canonical_clean = self._clean_canonical_name(canonical)
            if not canonical_clean:
                continue
                
            # Map all variants to canonical
            for variant in variants:
                variant_normalized = self._normalize_company_name(variant)
                if variant_normalized:  # Only map if not blacklisted
                    mappings[variant] = canonical_clean
        
        # Add ticker mappings
        for ticker, company in self.ticker_to_company.items():
            mappings[ticker] = company
        
        # Manual mappings for common variations
        manual_mappings = {
            'Apple Inc': 'Apple',
            'Apple Inc.': 'Apple', 
            'Microsoft Corp': 'Microsoft',
            'Microsoft Corp.': 'Microsoft',
            'Nvidia Corp': 'Nvidia',
            'Nvidia Corp.': 'Nvidia',
            'Alphabet Inc': 'Alphabet',
            'Alphabet Inc.': 'Alphabet',
            'Meta Platforms': 'Meta',
            'Tesla Inc': 'Tesla',
            'Tesla Inc.': 'Tesla',
            'Amazon.com': 'Amazon',
            'JPMorgan Chase & Co': 'JPMorgan Chase',
            'Goldman Sachs Group': 'Goldman Sachs',
            'Bank of America Corp': 'Bank of America',
            'Taiwan Semiconductor Manufacturing Company': 'Taiwan Semiconductor',
            'Taiwan Semiconductor Manufacturing Co Ltd': 'Taiwan Semiconductor',
            'Taiwan Semiconductor Manufacturing Co': 'Taiwan Semiconductor',
        }
        
        mappings.update(manual_mappings)
        
        return mappings
    
    def _clean_canonical_name(self, name: str) -> str:
        """Clean canonical name for final use"""
        if not name:
            return ""
            
        # Skip blacklisted terms
        if name.lower().strip() in self.blacklist_entities:
            return ""
            
        # Keep original capitalization but clean up
        cleaned = name.strip()
        
        # Fix common capitalization issues
        cleaned = re.sub(r'\bAi\b', 'AI', cleaned)
        cleaned = re.sub(r'\bIot\b', 'IoT', cleaned)
        cleaned = re.sub(r'\bApi\b', 'API', cleaned)
        cleaned = re.sub(r'\bUs\b', 'US', cleaned)
        cleaned = re.sub(r'\bUk\b', 'UK', cleaned)
        
        return cleaned
    
    def apply_deduplication(self, results: List[Dict]) -> List[Dict]:
        """Apply deduplication to all relationships and events"""
        deduplicated_results = []
        
        for result in results:
            new_result = result.copy()
            
            # Deduplicate relationships
            new_relationships = []
            for rel in result.get('relationships', []):
                source = self._map_entity(rel.get('source_entity', ''))
                target = self._map_entity(rel.get('target_entity', ''))
                
                # Only keep if both entities are valid after mapping
                if source and target and source != target:
                    new_rel = rel.copy()
                    new_rel['source_entity'] = source
                    new_rel['target_entity'] = target
                    new_relationships.append(new_rel)
            
            # Deduplicate events
            new_events = []
            for event in result.get('events', []):
                primary_entity = self._map_entity(event.get('primary_entity', ''))
                
                # Only keep if entity is valid after mapping
                if primary_entity:
                    new_event = event.copy()
                    new_event['primary_entity'] = primary_entity
                    new_events.append(new_event)
            
            new_result['relationships'] = new_relationships
            new_result['events'] = new_events
            new_result['original_relationship_count'] = len(result.get('relationships', []))
            new_result['original_event_count'] = len(result.get('events', []))
            new_result['deduplicated_relationship_count'] = len(new_relationships)
            new_result['deduplicated_event_count'] = len(new_events)
            
            deduplicated_results.append(new_result)
        
        return deduplicated_results
    
    def _map_entity(self, entity: str) -> str:
        """Map entity to canonical form"""
        if not entity or not entity.strip():
            return ""
            
        entity = entity.strip()
        
        # Direct mapping
        if entity in self.canonical_mappings:
            return self.canonical_mappings[entity]
        
        # Check if it's a blacklisted term
        if entity.lower() in self.blacklist_entities:
            return ""
        
        # Return as-is if no mapping found and not blacklisted
        return entity
    
    def remove_duplicate_relationships(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships within each article"""
        for result in results:
            relationships = result.get('relationships', [])
            if not relationships:
                continue
                
            # Create unique relationships based on source-relation-target
            seen_relationships = set()
            unique_relationships = []
            
            for rel in relationships:
                source = rel.get('source_entity', '')
                relation_type = rel.get('relationship_type', '')
                target = rel.get('target_entity', '')
                
                # Create a signature for deduplication
                signature = (source.lower(), relation_type.lower(), target.lower())
                
                if signature not in seen_relationships:
                    seen_relationships.add(signature)
                    unique_relationships.append(rel)
            
            result['relationships'] = unique_relationships
        
        return results

def analyze_deduplication_impact(original_results: List[Dict], 
                               deduplicated_results: List[Dict]) -> Dict:
    """Analyze the impact of deduplication"""
    original_relationships = sum(len(r.get('relationships', [])) for r in original_results)
    original_events = sum(len(r.get('events', [])) for r in original_results)
    
    final_relationships = sum(len(r.get('relationships', [])) for r in deduplicated_results)
    final_events = sum(len(r.get('events', [])) for r in deduplicated_results)
    
    # Count unique entities after deduplication
    unique_entities = set()
    for result in deduplicated_results:
        for rel in result.get('relationships', []):
            unique_entities.add(rel.get('source_entity', ''))
            unique_entities.add(rel.get('target_entity', ''))
        for event in result.get('events', []):
            unique_entities.add(event.get('primary_entity', ''))
    
    # Remove empty entities
    unique_entities.discard('')
    
    return {
        'original_relationships': original_relationships,
        'final_relationships': final_relationships,
        'relationships_removed': original_relationships - final_relationships,
        'relationships_retention_rate': final_relationships / original_relationships if original_relationships > 0 else 0,
        
        'original_events': original_events,
        'final_events': final_events,
        'events_removed': original_events - final_events,
        'events_retention_rate': final_events / original_events if original_events > 0 else 0,
        
        'unique_entities_after_dedup': len(unique_entities),
        'unique_entities_list': sorted(list(unique_entities))
    }

def main():
    """Main deduplication pipeline"""
    
    # Load original extraction results
    print("Loading extraction results...")
    with open("extracted_relationships_events_FULL.json", "r", encoding="utf-8") as f:
        original_results = json.load(f)
    
    # Load entity analysis
    print("Loading entity analysis...")
    with open("entity_analysis/company_variations.json", "r", encoding="utf-8") as f:
        company_variations = json.load(f)
    
    # Initialize deduplicator
    deduplicator = EntityDeduplicator()
    
    # Build canonical mappings
    print("Building canonical mappings...")
    deduplicator.canonical_mappings = deduplicator.build_canonical_mappings(company_variations)
    
    # Apply deduplication
    print("Applying deduplication...")
    deduplicated_results = deduplicator.apply_deduplication(original_results)
    
    # Remove duplicate relationships
    print("Removing duplicate relationships...")
    deduplicated_results = deduplicator.remove_duplicate_relationships(deduplicated_results)
    
    # Analyze impact
    print("Analyzing deduplication impact...")
    impact_analysis = analyze_deduplication_impact(original_results, deduplicated_results)
    
    # Save results
    output_dir = Path("deduplicated_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save deduplicated extraction results
    with open(output_dir / "deduplicated_relationships_events.json", "w", encoding="utf-8") as f:
        json.dump(deduplicated_results, f, indent=2, ensure_ascii=False)
    
    # Save canonical mappings
    with open(output_dir / "entity_mappings.json", "w", encoding="utf-8") as f:
        json.dump(deduplicator.canonical_mappings, f, indent=2, ensure_ascii=False)
    
    # Save impact analysis
    with open(output_dir / "deduplication_impact.json", "w", encoding="utf-8") as f:
        json.dump(impact_analysis, f, indent=2, ensure_ascii=False)
    
    # Save clean entity list
    with open(output_dir / "clean_entities.json", "w", encoding="utf-8") as f:
        json.dump(impact_analysis['unique_entities_list'], f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*60)
    print("DEDUPLICATION SUMMARY")
    print("="*60)
    print(f"Original relationships: {impact_analysis['original_relationships']:,}")
    print(f"Final relationships: {impact_analysis['final_relationships']:,}")
    print(f"Relationships removed: {impact_analysis['relationships_removed']:,}")
    print(f"Retention rate: {impact_analysis['relationships_retention_rate']:.1%}")
    print()
    print(f"Original events: {impact_analysis['original_events']:,}")
    print(f"Final events: {impact_analysis['final_events']:,}")
    print(f"Events removed: {impact_analysis['events_removed']:,}")
    print(f"Retention rate: {impact_analysis['events_retention_rate']:.1%}")
    print()
    print(f"Unique entities after deduplication: {impact_analysis['unique_entities_after_dedup']:,}")
    print(f"Entity mappings created: {len(deduplicator.canonical_mappings):,}")
    print("="*60)
    print(f"Results saved to: {output_dir}/")

if __name__ == "__main__":
    main()