"""
Direct Neo4j Database Loader
Connect directly to Neo4j instance and load financial knowledge graph
"""

import json
from pathlib import Path
from typing import Dict, List
import re
import os

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j driver not installed. Install with: pip install neo4j")

class Neo4jDirectLoader:
    def __init__(self, uri: str, username: str, password: str):
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print(f"Connected to Neo4j at {uri}")
    
    def close(self):
        if self.driver:
            self.driver.close()
    
    def clear_database(self):
        """Clear all existing data"""
        with self.driver.session() as session:
            print("Clearing existing data...")
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def create_constraints(self):
        """Create constraints and indexes"""
        with self.driver.session() as session:
            print("Creating constraints and indexes...")
            
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
                "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE", 
                "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE"
            ]
            
            indexes = [
                "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
                "CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.name)",
                "CREATE INDEX event_type IF NOT EXISTS FOR (n:Event) ON (n.event_type)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
            
            print("Constraints and indexes created.")
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string for Neo4j"""
        if not text:
            return ""
        text = str(text).strip()
        return text[:1000] if len(text) > 1000 else text
    
    def create_entity_nodes(self, entities: Dict[str, Dict]):
        """Create all entity nodes in batches"""
        with self.driver.session() as session:
            print(f"Creating {len(entities)} entity nodes...")
            
            batch_size = 1000
            entity_list = list(entities.items())
            
            for i in range(0, len(entity_list), batch_size):
                batch = entity_list[i:i + batch_size]
                
                batch_data = []
                for entity_id, entity_data in batch:
                    batch_data.append({
                        'id': self.sanitize_string(entity_id),
                        'name': self.sanitize_string(entity_id),
                        'type': 'company',
                        'mention_count': entity_data.get('mentions', 0)
                    })
                
                query = """
                UNWIND $batch AS row
                CREATE (n:Entity:Company {
                    id: row.id,
                    name: row.name,
                    type: row.type,
                    mention_count: row.mention_count
                })
                """
                
                session.run(query, batch=batch_data)
                print(f"Created entities batch {i//batch_size + 1}/{(len(entity_list)-1)//batch_size + 1}")
    
    def create_relationships(self, relationships: List[Dict]):
        """Create relationship edges in batches"""
        with self.driver.session() as session:
            print(f"Creating {len(relationships)} relationships...")
            
            batch_size = 500
            
            for i in range(0, len(relationships), batch_size):
                batch = relationships[i:i + batch_size]
                
                batch_data = []
                for rel in batch:
                    batch_data.append({
                        'source': self.sanitize_string(rel['source']),
                        'target': self.sanitize_string(rel['target']),
                        'type': rel['rel_type'].upper().replace(' ', '_').replace('-', '_'),
                        'confidence': rel.get('confidence', 0.5),
                        'evidence': self.sanitize_string(rel.get('evidence', ''))[:500],
                        'direction': rel.get('direction', 'neutral'),
                        'financial_value': self.sanitize_string(rel.get('financial_value', '')),
                        'headline': self.sanitize_string(rel.get('headline', ''))[:200]
                    })
                
                query = """
                UNWIND $batch AS row
                MATCH (a:Entity {id: row.source})
                MATCH (b:Entity {id: row.target})
                CALL apoc.create.relationship(a, row.type, {
                    confidence: row.confidence,
                    evidence: row.evidence,
                    direction: row.direction,
                    financial_value: row.financial_value,
                    headline: row.headline
                }, b) YIELD rel
                RETURN count(rel)
                """
                
                fallback_query = """
                UNWIND $batch AS row
                MATCH (a:Entity {id: row.source})
                MATCH (b:Entity {id: row.target})
                CREATE (a)-[r:RELATED_TO]->(b)
                SET r.relationship_type = row.type,
                    r.confidence = row.confidence,
                    r.evidence = row.evidence,
                    r.direction = row.direction,
                    r.financial_value = row.financial_value,
                    r.headline = row.headline
                """
                
                try:
                    session.run(query, batch=batch_data)
                except Exception as e:
                    if "apoc" in str(e).lower():
                        session.run(fallback_query, batch=batch_data)
                    else:
                        print(f"Error in relationship batch {i//batch_size + 1}: {e}")
                        continue
                
                print(f"Created relationships batch {i//batch_size + 1}/{(len(relationships)-1)//batch_size + 1}")
    
    def create_events(self, events: List[Dict]):
        """Create event nodes and connections"""
        with self.driver.session() as session:
            print(f"Creating {len(events)} event nodes...")
            
            batch_size = 500
            
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                
                event_data = []
                for event in batch:
                    event_data.append({
                        'id': event['event_id'],
                        'name': self.sanitize_string(event['description'])[:100],
                        'type': 'event',
                        'event_type': event['event_type'],
                        'description': self.sanitize_string(event['description']),
                        'confidence': event.get('confidence', 0.5),
                        'evidence': self.sanitize_string(event.get('evidence', ''))[:500],
                        'financial_impact': self.sanitize_string(event.get('financial_impact', '')),
                        'market_direction': event.get('market_direction', 'neutral'),
                        'scale': event.get('scale', 'moderate'),
                        'headline': self.sanitize_string(event.get('headline', ''))[:200]
                    })
                
                event_query = """
                UNWIND $batch AS row
                CREATE (e:Entity:Event {
                    id: row.id,
                    name: row.name,
                    type: row.type,
                    event_type: row.event_type,
                    description: row.description,
                    confidence: row.confidence,
                    evidence: row.evidence,
                    financial_impact: row.financial_impact,
                    market_direction: row.market_direction,
                    scale: row.scale,
                    headline: row.headline
                })
                """
                
                session.run(event_query, batch=event_data)
                
                connection_data = []
                for event in batch:
                    if event.get('primary_entity'):
                        connection_data.append({
                            'event_id': event['event_id'],
                            'entity_id': self.sanitize_string(event['primary_entity']),
                            'event_type': event['event_type'],
                            'confidence': event.get('confidence', 0.5)
                        })
                
                if connection_data:
                    connection_query = """
                    UNWIND $batch AS row
                    MATCH (entity:Entity {id: row.entity_id})
                    MATCH (event:Event {id: row.event_id})
                    CREATE (entity)-[r:INVOLVED_IN_EVENT]->(event)
                    SET r.event_type = row.event_type,
                        r.confidence = row.confidence
                    """
                    
                    session.run(connection_query, batch=connection_data)
                
                print(f"Created events batch {i//batch_size + 1}/{(len(events)-1)//batch_size + 1}")
    
    def load_financial_graph(self, deduplicated_data: List[Dict]):
        """Load complete financial knowledge graph"""
        print("Starting knowledge graph load...")
        
        self.clear_database()
        
        self.create_constraints()
        
        entities = {}
        relationships = []
        events = []
        
        print("Processing data...")
        for article_idx, article in enumerate(deduplicated_data):
            headline = article.get('headline', '')
            
            for rel in article.get('relationships', []):
                source = rel.get('source_entity', '').strip()
                target = rel.get('target_entity', '').strip()
                rel_type = rel.get('relationship_type', '').strip()
                
                if source and target and rel_type:
                    entities[source] = {'mentions': entities.get(source, {}).get('mentions', 0) + 1}
                    entities[target] = {'mentions': entities.get(target, {}).get('mentions', 0) + 1}
                    
                    relationships.append({
                        'source': source,
                        'target': target,
                        'rel_type': rel_type,
                        'confidence': rel.get('confidence', 0.5),
                        'evidence': rel.get('evidence', ''),
                        'direction': rel.get('metadata', {}).get('direction', 'neutral'),
                        'financial_value': rel.get('metadata', {}).get('financial_value', ''),
                        'headline': headline
                    })
            
            for event_idx, event in enumerate(article.get('events', [])):
                event_type = event.get('event_type', '').strip()
                primary_entity = event.get('primary_entity', '').strip()
                
                if event_type and primary_entity:
                    entities[primary_entity] = {'mentions': entities.get(primary_entity, {}).get('mentions', 0) + 1}
                    
                    events.append({
                        'event_id': f"EVENT_{article_idx}_{event_idx}_{event_type}",
                        'event_type': event_type,
                        'primary_entity': primary_entity,
                        'description': event.get('event_description', ''),
                        'confidence': event.get('confidence', 0.5),
                        'evidence': event.get('evidence', ''),
                        'financial_impact': event.get('metadata', {}).get('financial_impact', ''),
                        'market_direction': event.get('metadata', {}).get('market_direction', 'neutral'),
                        'scale': event.get('metadata', {}).get('scale', 'moderate'),
                        'headline': headline
                    })
        
        self.create_entity_nodes(entities)
        self.create_relationships(relationships)
        self.create_events(events)
        
        print(f"\nKnowledge graph loaded successfully!")
        print(f"Entities: {len(entities)}")
        print(f"Relationships: {len(relationships)}")
        print(f"Events: {len(events)}")
    
    def run_sample_queries(self):
        """Run some sample queries to verify the data"""
        with self.driver.session() as session:
            print("\n" + "="*50)
            print("SAMPLE QUERIES")
            print("="*50)
            
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            print("\nNode counts:")
            for record in result:
                print(f"  {record['labels']}: {record['count']}")
            
            result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as relationship_type, count(r) as count 
                ORDER BY count DESC LIMIT 10
            """)
            print("\nTop relationship types:")
            for record in result:
                print(f"  {record['relationship_type']}: {record['count']}")
            
            result = session.run("""
                MATCH (n:Company) 
                RETURN n.name as name, n.mention_count as mentions 
                ORDER BY n.mention_count DESC LIMIT 10
            """)
            print("\nMost mentioned companies:")
            for record in result:
                print(f"  {record['name']}: {record['mentions']} mentions")

def main():
    """Main function"""
    
    if not NEO4J_AVAILABLE:
        print("Please install Neo4j driver: pip install neo4j")
        return
    
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    
    input_file = "deduplicated_results/deduplicated_relationships_events.json"
    if not Path(input_file).exists():
        print(f"Error: {input_file} not found.")
        print("Run the deduplication script first.")
        return
    
    try:
        loader = Neo4jDirectLoader(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        
        print("Loading deduplicated data...")
        with open(input_file, 'r', encoding='utf-8') as f:
            deduplicated_data = json.load(f)
        
        print(f"Loaded {len(deduplicated_data)} articles")
        
        loader.load_financial_graph(deduplicated_data)
        
        loader.run_sample_queries()
        
        print("\n" + "="*50)
        print("SUCCESS!")
        print("="*50)
        print("Your financial knowledge graph is now loaded in Neo4j!")
        print("Open Neo4j Browser and try these queries:")
        print()
        print("// Find Apple's relationships")
        print("MATCH (a:Company {name: 'Apple'})-[r]->(b) RETURN a, r, b LIMIT 20")
        print()
        print("// Find competitive relationships")
        print("MATCH (a)-[r]-(b) WHERE r.relationship_type = 'COMPETES_WITH' RETURN a, r, b LIMIT 10")
        print()
        print("// Explore the graph visually")
        print("MATCH (n) RETURN n LIMIT 50")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your Neo4j database is running and credentials are correct.")
    
    finally:
        if 'loader' in locals():
            loader.close()

if __name__ == "__main__":
    main()