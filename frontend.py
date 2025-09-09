"""
Comprehensive Knowledge Graph Query System
Advanced NLP-powered system that can handle almost any question about your knowledge graph
"""

from flask import Flask, render_template_string, request, jsonify
from neo4j import GraphDatabase
import re
import json
import spacy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import os
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

app = Flask(__name__)

try:
    nlp = spacy.load("en_core_web_sm")
    NLP_AVAILABLE = True
    print("✅ spaCy NLP model loaded")
except OSError:
    NLP_AVAILABLE = False
    print("⚠️ spaCy not available. Install with: python -m spacy download en_core_web_sm")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print("✅ Connected to Neo4j successfully!")
except Exception as e:
    print(f"❌ Failed to connect to Neo4j: {e}")
    driver = None

class QueryType(Enum):
    RELATIONSHIP = "relationship"
    COMPARISON = "comparison"
    AGGREGATION = "aggregation"
    EVENTS = "events"
    NETWORK_ANALYSIS = "network_analysis"
    FINANCIAL = "financial"
    TEMPORAL = "temporal"
    EXPLORATORY = "exploratory"

@dataclass
class QueryIntent:
    query_type: QueryType
    entities: List[str]
    relationships: List[str]
    properties: List[str]
    modifiers: List[str]
    confidence: float

class ComprehensiveQueryTranslator:
    """Advanced query translator using NLP and pattern matching"""
    
    def __init__(self):
        self.entity_cache = set()
        self.relationship_types = set()
        self.event_types = set()
        self._load_graph_schema()
        
        self.query_patterns = {
            QueryType.RELATIONSHIP: [
                {
                    'patterns': [
                        r'(?:what|which|who).*(?:relationship|connect|link|relate).*(?:with|to|between).*?(\w+)',
                        r'(?:how|what).*(\w+).*(?:connected|related|linked).*(?:to|with).*?(\w+)',
                        r'(?:find|show|list).*(?:relationship|connection).*(?:between|with).*?(\w+)',
                        r'(\w+).*(?:partner|compete|collaborate|work with|ally).*(?:with|to).*?(\w+)',
                    ],
                    'cypher_template': '''
                        MATCH (e1)-[r]-(e2)
                        WHERE ({entity_filters})
                        RETURN e1.name as entity1, type(r) as relationship, 
                               e2.name as entity2, r.confidence as confidence,
                               r.evidence as evidence
                        ORDER BY r.confidence DESC
                        LIMIT 20
                    '''
                },
                
                {
                    'patterns': [
                        r'(?:all|every).*(?:partner|collaboration|alliance).*(?:of|with).*?(\w+)',
                        r'(\w+).*(?:partner|collaborate|work with)',
                        r'(?:who|what|which).*(?:partner|collaborate).*(?:with|to).*?(\w+)'
                    ],
                    'cypher_template': '''
                        MATCH (company:Company)-[r]-(partner:Company)
                        WHERE toLower(company.name) CONTAINS toLower($entity)
                        AND (r.relationship_type CONTAINS 'PARTNER' OR 
                             r.relationship_type CONTAINS 'COLLABORAT' OR
                             r.relationship_type CONTAINS 'ALLIANCE' OR
                             type(r) CONTAINS 'PARTNER')
                        RETURN partner.name as partner, r.relationship_type as type,
                               r.confidence as confidence, r.financial_value as value
                        ORDER BY r.confidence DESC
                    '''
                }
            ],
            
            QueryType.COMPARISON: [
                {
                    'patterns': [
                        r'compar.*(\w+).*(?:with|to|versus|vs).*?(\w+)',
                        r'(\w+).*(?:versus|vs|compared to).*?(\w+)',
                        r'(?:difference|differences).*(?:between).*?(\w+).*(?:and).*?(\w+)',
                        r'(?:which|who).*(?:better|stronger|bigger|larger).*(\w+).*(?:or).*?(\w+)'
                    ],
                    'cypher_template': '''
                        MATCH (e1:Company)-[r1]-(others1)
                        MATCH (e2:Company)-[r2]-(others2)  
                        WHERE toLower(e1.name) CONTAINS toLower($entity1)
                        AND toLower(e2.name) CONTAINS toLower($entity2)
                        WITH e1, e2, count(r1) as connections1, count(r2) as connections2
                        RETURN e1.name as company1, connections1,
                               e2.name as company2, connections2,
                               CASE WHEN connections1 > connections2 THEN e1.name 
                                    ELSE e2.name END as more_connected
                    '''
                }
            ],
            
            QueryType.AGGREGATION: [
                {
                    'patterns': [
                        r'(?:most|top|highest|largest).*(?:connect|relationship|popular)',
                        r'(?:which|what).*(?:company|companies).*(?:most|highest|top)',
                        r'(?:rank|ranking|list).*(?:by|according to).*(?:connection|relationship)',
                        r'(?:biggest|largest|major).*(?:player|company|companies)'
                    ],
                    'cypher_template': '''
                        MATCH (c:Company)-[r]-()
                        RETURN c.name as company, 
                               count(r) as total_relationships,
                               c.mention_count as mentions,
                               labels(c) as types
                        ORDER BY total_relationships DESC
                        LIMIT 15
                    '''
                },
                
                {
                    'patterns': [
                        r'(?:how many|count|number of).*(?:relationship|connection|partner)',
                        r'(?:total|sum).*(?:relationship|connection)',
                        r'(?:statistics|stats).*(?:about|on).*(?:graph|network)'
                    ],
                    'cypher_template': '''
                        MATCH (n)-[r]-(m)
                        RETURN count(DISTINCT n) as total_entities,
                               count(DISTINCT r) as total_relationships,
                               count(DISTINCT type(r)) as relationship_types
                    '''
                }
            ],
            
            QueryType.EVENTS: [
                {
                    'patterns': [
                        r'(?:event|events|happening|occurred).*(?:involving|with|about).*?(\w+)',
                        r'(?:what|which).*(?:event|news|announcement).*?(\w+)',
                        r'(?:merger|acquisition|M&A|deal|transaction).*?(\w+)',
                        r'(\w+).*(?:acquire|merge|bought|purchase|deal)'
                    ],
                    'cypher_template': '''
                        MATCH (e:Event)-[r]-(company:Company)
                        WHERE toLower(company.name) CONTAINS toLower($entity)
                        RETURN e.description as event, e.event_type as type,
                               company.name as company, e.confidence as confidence,
                               e.financial_impact as impact, e.market_direction as direction
                        ORDER BY e.confidence DESC
                        LIMIT 10
                    '''
                },
                
                {
                    'patterns': [
                        r'(?:all|recent|latest).*(?:merger|acquisition|M&A|deal)',
                        r'(?:show|list|find).*(?:event|transaction|deal)',
                        r'(?:market|financial).*(?:event|news|announcement)'
                    ],
                    'cypher_template': '''
                        MATCH (e:Event)-[r]-(c:Company)
                        WHERE e.event_type IN ['MERGER', 'ACQUISITION', 'M&A', 'DEAL', 'TRANSACTION']
                        RETURN e.description as event, c.name as company,
                               e.event_type as type, e.financial_impact as value,
                               e.confidence as confidence
                        ORDER BY e.confidence DESC
                        LIMIT 15
                    '''
                }
            ],
            
            QueryType.NETWORK_ANALYSIS: [
                {
                    'patterns': [
                        r'(?:network|graph).*(?:analysis|structure|pattern)',
                        r'(?:central|important|influential).*(?:company|player|node)',
                        r'(?:hub|center|core).*(?:of|in).*(?:network|industry)',
                        r'(?:cluster|group|community).*(?:of|in).*(?:companies|industry)'
                    ],
                    'cypher_template': '''
                        MATCH (c:Company)-[r]-(other:Company)
                        WITH c, count(r) as degree, collect(DISTINCT type(r)) as rel_types
                        WHERE degree > 3
                        RETURN c.name as company, degree as connections,
                               size(rel_types) as relationship_diversity,
                               rel_types as relationship_types
                        ORDER BY degree DESC, relationship_diversity DESC
                        LIMIT 10
                    '''
                }
            ],
            
            QueryType.FINANCIAL: [
                {
                    'patterns': [
                        r'(?:financial|money|revenue|profit|value).*(?:impact|effect|relation)',
                        r'(?:investment|funding|valuation).*?(\w+)',
                        r'(?:worth|value|price).*(?:of|for).*?(\w+)',
                        r'(\w+).*(?:invest|fund|value|worth)'
                    ],
                    'cypher_template': '''
                        MATCH (c:Company)-[r]-(other)
                        WHERE toLower(c.name) CONTAINS toLower($entity)
                        AND (r.financial_value IS NOT NULL OR r.relationship_type CONTAINS 'INVEST')
                        RETURN c.name as company, other.name as related,
                               r.relationship_type as type, r.financial_value as value,
                               r.confidence as confidence
                        ORDER BY r.confidence DESC
                    '''
                }
            ],
            
            QueryType.EXPLORATORY: [
                {
                    'patterns': [
                        r'(?:tell me|what|anything).*(?:about|regarding).*?(\w+)',
                        r'(?:information|info|details).*(?:on|about).*?(\w+)',
                        r'(?:explore|discover|find out).*(?:about).*?(\w+)',
                        r'(\w+).*(?:overview|summary|profile)'
                    ],
                    'cypher_template': '''
                        MATCH (c:Company)-[r]-(other)
                        WHERE toLower(c.name) CONTAINS toLower($entity)
                        WITH c, type(r) as rel_type, count(r) as count
                        RETURN c.name as company, rel_type as relationship_type, count
                        ORDER BY count DESC
                        LIMIT 10
                    '''
                }
            ]
        }
    
    def _load_graph_schema(self):
        """Load available entities and relationships from the graph"""
        if not driver:
            return
            
        try:
            with driver.session() as session:
                result = session.run("MATCH (c:Company) RETURN c.name as name LIMIT 100")
                self.entity_cache.update([record['name'].lower() for record in result])
                
                result = session.run("MATCH ()-[r]-() RETURN DISTINCT type(r) as rel_type")
                self.relationship_types.update([record['rel_type'] for record in result])
                
                result = session.run("MATCH (e:Event) RETURN DISTINCT e.event_type as event_type")
                self.event_types.update([record['event_type'] for record in result if record['event_type']])
                
        except Exception as e:
            print(f"Warning: Could not load graph schema: {e}")
    
    def extract_entities_nlp(self, text: str) -> List[str]:
        """Extract entities using spaCy NLP"""
        if not NLP_AVAILABLE:
            return self.extract_entities_simple(text)
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:
                entities.append(ent.text)
        
        for token in doc:
            if token.pos_ == 'PROPN' and token.text.isalpha():
                entities.append(token.text)
        
        known_entities = []
        for entity in entities:
            if entity.lower() in self.entity_cache:
                known_entities.append(entity)
        
        return known_entities if known_entities else entities
    
    def extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction fallback"""
        companies = [
            'Apple', 'Microsoft', 'Google', 'Amazon', 'Meta', 'Tesla', 'Netflix',
            'Salesforce', 'Oracle', 'IBM', 'Intel', 'AMD', 'NVIDIA', 'Samsung',
            'Adobe', 'Uber', 'Airbnb', 'Twitter', 'LinkedIn', 'Spotify', 'Zoom',
            'Facebook', 'Alphabet', 'Berkshire Hathaway', 'JPMorgan', 'Visa'
        ]
        
        found = []
        text_lower = text.lower()
        for company in companies:
            if company.lower() in text_lower:
                found.append(company)
        
        words = text.split()
        for word in words:
            if word.isalpha() and word[0].isupper() and len(word) > 2:
                if word not in found:
                    found.append(word)
        
        return found
    
    def analyze_intent(self, question: str) -> QueryIntent:
        """Analyze question intent using NLP and patterns"""
        question_lower = question.lower()
        
        entities = self.extract_entities_nlp(question)
        
        for query_type, pattern_groups in self.query_patterns.items():
            for pattern_group in pattern_groups:
                for pattern in pattern_group['patterns']:
                    if re.search(pattern, question_lower):
                        return QueryIntent(
                            query_type=query_type,
                            entities=entities,
                            relationships=[],
                            properties=[],
                            modifiers=[],
                            confidence=0.85
                        )
        
        return QueryIntent(
            query_type=QueryType.EXPLORATORY,
            entities=entities,
            relationships=[],
            properties=[],
            modifiers=[],
            confidence=0.6
        )
    
    def generate_cypher(self, intent: QueryIntent, question: str) -> Dict[str, Any]:
        """Generate Cypher query based on intent"""
        question_lower = question.lower()
        
        for pattern_group in self.query_patterns[intent.query_type]:
            for pattern in pattern_group['patterns']:
                match = re.search(pattern, question_lower)
                if match:
                    cypher = pattern_group['cypher_template']
                    params = {}
                    
                    if match.groups():
                        if len(match.groups()) >= 1:
                            params['entity'] = match.group(1).title()
                        if len(match.groups()) >= 2:
                            params['entity1'] = match.group(1).title()
                            params['entity2'] = match.group(2).title()
                    
                    if not params and intent.entities:
                        params['entity'] = intent.entities[0]
                    
                    if '{entity_filters}' in cypher and intent.entities:
                        filters = []
                        for i, entity in enumerate(intent.entities[:2]):
                            filters.append(f"toLower(e{i+1}.name) CONTAINS toLower('{entity}')")
                        entity_filter = ' OR '.join(filters)
                        cypher = cypher.replace('{entity_filters}', entity_filter)
                    
                    return {
                        'cypher': cypher,
                        'params': params,
                        'intent': intent.query_type.value,
                        'confidence': intent.confidence
                    }
        
        return {
            'cypher': '''
                MATCH (n)-[r]-(m)
                RETURN n.name as entity1, type(r) as relationship, 
                       m.name as entity2, r.confidence as confidence
                ORDER BY r.confidence DESC
                LIMIT 20
            ''',
            'params': {},
            'intent': 'general',
            'confidence': 0.3
        }
    
    def translate_question(self, question: str) -> Dict[str, Any]:
        """Main translation method"""
        intent = self.analyze_intent(question)
        
        query_info = self.generate_cypher(intent, question)
        
        answer_templates = {
            QueryType.RELATIONSHIP: f"Relationships found for your query:",
            QueryType.COMPARISON: f"Comparison results:",
            QueryType.AGGREGATION: f"Analysis results:",
            QueryType.EVENTS: f"Events found:",
            QueryType.NETWORK_ANALYSIS: f"Network analysis results:",
            QueryType.FINANCIAL: f"Financial information:",
            QueryType.EXPLORATORY: f"Information found:"
        }
        
        return {
            'query': query_info['cypher'],
            'params': query_info['params'],
            'answer': answer_templates.get(intent.query_type, "Results from knowledge graph:"),
            'intent': query_info['intent'],
            'confidence': query_info['confidence'],
            'entities': intent.entities
        }

translator = ComprehensiveQueryTranslator()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Knowledge Graph Query</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg,
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.15);
            width: 100%;
            max-width: 1000px;
            height: 700px;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg,
            color: white;
            padding: 25px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 28px;
            margin-bottom: 8px;
            font-weight: 700;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 16px;
        }
        
        .stats {
            background: rgba(255,255,255,0.1);
            margin: 15px 0;
            padding: 10px;
            border-radius: 10px;
            font-size: 14px;
        }
        
        .messages {
            flex: 1;
            padding: 25px;
            overflow-y: auto;
            background:
        }
        
        .message {
            margin-bottom: 20px;
            padding: 18px;
            border-radius: 12px;
            max-width: 85%;
            line-height: 1.5;
        }
        
        .user-message {
            background: linear-gradient(135deg,
            color: white;
            margin-left: auto;
        }
        
        .bot-message {
            background: white;
            border: 1px solid
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .loading {
            background:
            color:
            font-style: italic;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid
            border-top: 2px solid
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .intent-badge {
            background:
            color:
            padding: 4px 8px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 500;
            margin-bottom: 10px;
            display: inline-block;
        }
        
        .confidence-bar {
            background:
            height: 4px;
            border-radius: 2px;
            margin: 5px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg,
            transition: width 0.3s ease;
        }
        
        .result-item {
            background:
            border: 1px solid
            padding: 15px;
            margin: 8px 0;
            border-radius: 8px;
            font-size: 14px;
        }
        
        .result-item:hover {
            background:
            border-color:
        }
        
        .result-key {
            font-weight: 600;
            color:
            text-transform: capitalize;
        }
        
        .result-value {
            color:
            margin-left: 8px;
        }
        
        .query-box {
            background:
            color:
            padding: 15px;
            border-radius: 8px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 13px;
            margin: 15px 0;
            overflow-x: auto;
            border-left: 4px solid
        }
        
        .query-label {
            color:
            font-size: 11px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .input-area {
            padding: 25px;
            background: white;
            border-top: 1px solid
        }
        
        .input-container {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
            flex: 1;
            padding: 15px 20px;
            border: 2px solid
            border-radius: 30px;
            font-size: 16px;
            outline: none;
            transition: all 0.2s;
        }
        
            border-color:
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
            padding: 15px 30px;
            background: linear-gradient(135deg,
            color: white;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.2s;
            min-width: 120px;
        }
        
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
            background:
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .samples {
            margin-bottom: 20px;
        }
        
        .samples-title {
            font-weight: 600;
            margin-bottom: 12px;
            color:
            font-size: 14px;
        }
        
        .sample-question {
            background: linear-gradient(135deg,
            color:
            padding: 10px 15px;
            margin: 6px;
            border-radius: 20px;
            font-size: 13px;
            cursor: pointer;
            display: inline-block;
            transition: all 0.2s;
            border: 1px solid transparent;
        }
        
        .sample-question:hover {
            background: linear-gradient(135deg,
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(29, 78, 216, 0.2);
        }
        
        .no-results {
            text-align: center;
            color:
            font-style: italic;
            padding: 40px;
            background:
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .error-message {
            background:
            color:
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Advanced Knowledge Graph</h1>
            <p>Ask complex questions about companies, relationships, events, and financial data</p>
            <div class="stats" id="graphStats">
                <div>Loading graph statistics...</div>
            </div>
        </div>
        
        <div class="messages" id="messages">
            <div class="message bot-message">
                <strong>🚀 Welcome to the Advanced Knowledge Graph Query System!</strong>
                <p>I can understand complex questions about your financial knowledge graph using advanced NLP. Try asking about relationships, comparisons, events, or network analysis.</p>
                
                <div class="samples">
                    <div class="samples-title">💡 Try these advanced questions:</div>
                    <div class="sample-question" onclick="askSample('Compare Apple and Microsoft in terms of partnerships and competitive relationships')">
                        Compare Apple and Microsoft partnerships
                    </div>
                    <div class="sample-question" onclick="askSample('What are the most influential companies in the network based on their connections?')">
                        Most influential companies in network
                    </div>
                    <div class="sample-question" onclick="askSample('Show me all merger and acquisition events with financial impact')">
                        M&A events with financial impact
                    </div>
                    <div class="sample-question" onclick="askSample('Which companies have investment relationships and what are the values?')">
                        Investment relationships and values
                    </div>
                    <div class="sample-question" onclick="askSample('Find companies that are both competitors and partners in different contexts')">
                        Complex competitor-partner relationships
                    </div>
                    <div class="sample-question" onclick="askSample('What events involving Tesla have the highest market impact?')">
                        Tesla events with market impact
                    </div>
                </div>
            </div>
        </div>
        
        <div class="input-area">
            <div class="input-container">
                <input 
                    type="text" 
                    id="questionInput" 
                    placeholder="Ask any complex question about the knowledge graph..."
                    onkeypress="if(event.key==='Enter') askQuestion()"
                >
                <button id="askButton" onclick="askQuestion()">Ask</button>
            </div>
        </div>
    </div>

    <script>
        // Load graph stats on startup
        loadGraphStats();
        
        async function loadGraphStats() {
            try {
                const response = await fetch('/stats');
                const stats = await response.json();
                document.getElementById('graphStats').innerHTML = 
                    `📊 ${stats.total_entities || 0} entities • ${stats.total_relationships || 0} relationships • ${stats.relationship_types || 0} types`;
            } catch (error) {
                document.getElementById('graphStats').innerHTML = '📊 Graph statistics unavailable';
            }
        }
        
        function askSample(question) {
            document.getElementById('questionInput').value = question;
            askQuestion();
        }
        
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const button = document.getElementById('askButton');
            const question = input.value.trim();
            
            if (!question) return;
            
            // Add user message
            addMessage(question, 'user-message');
            
            // Show loading
            const loadingId = addMessage('<div class="spinner"></div>Analyzing question and querying knowledge graph...', 'loading');
            
            // Disable input
            input.disabled = true;
            button.disabled = true;
            input.value = '';
            
            try {
                const response = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                // Remove loading message
                document.getElementById(loadingId).remove();
                
                // Add response
                addBotResponse(data);
                
            } catch (error) {
                document.getElementById(loadingId).innerHTML = 
                    '<div class="error-message">❌ Error: Could not connect to knowledge graph. Please check your Neo4j connection.</div>';
                document.getElementById(loadingId).className = 'message bot-message';
            }
            
            // Re-enable input
            input.disabled = false;
            button.disabled = false;
            input.focus();
        }
        
        function addMessage(text, className) {
            const messages = document.getElementById('messages');
            const messageId = 'msg_' + Date.now();
            const div = document.createElement('div');
            div.id = messageId;
            div.className = 'message ' + className;
            div.innerHTML = text;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
            return messageId;
        }
        
        function addBotResponse(data) {
            const messages = document.getElementById('messages');
            const div = document.createElement('div');
            div.className = 'message bot-message';
            
            let html = '';
            
            // Intent and confidence
            if (data.intent) {
                html += `<div class="intent-badge">Query Type: ${data.intent}</div>`;
            }
            
            if (data.confidence) {
                html += `<div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${data.confidence * 100}%"></div>
                </div>
                <div style="font-size: 12px; color: #6b7280; margin-bottom: 10px;">
                    Confidence: ${Math.round(data.confidence * 100)}%
                </div>`;
            }
            
            html += '<strong>' + data.answer + '</strong>';
            
            // Results
            if (data.results && data.results.length > 0) {
                html += `<div style="margin: 15px 0; color: #6b7280; font-size: 14px;">Found ${data.results.length} results:</div>`;
                
                data.results.forEach(result => {
                    html += '<div class="result-item">';
                    Object.keys(result).forEach(key => {
                        if (result[key] !== null && result[key] !== '') {
                            html += `<div><span class="result-key">${key}:</span><span class="result-value">${result[key]}</span></div>`;
                        }
                    });
                    html += '</div>';
                });
            } else {
                html += '<div class="no-results">🔍 No results found. Try rephrasing your question or using different company names.</div>';
            }
            
            // Show Cypher query
            if (data.cypher) {
                html += '<div class="query-box">';
                html += '<div class="query-label">Generated Cypher Query:</div>';
                html += data.cypher;
                html += '</div>';
            }
            
            // Entities detected
            if (data.entities && data.entities.length > 0) {
                html += `<div style="margin-top: 10px; font-size: 12px; color: #6b7280;">
                    🎯 Detected entities: ${data.entities.join(', ')}
                </div>`;
            }
            
            div.innerHTML = html;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Focus input on load
        document.getElementById('questionInput').focus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/query', methods=['POST'])
def query():
    """Handle complex questions with advanced NLP"""
    if not driver:
        return jsonify({'error': 'Not connected to Neo4j'}), 500
    
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        translation = translator.translate_question(question)
        
        with driver.session() as session:
            result = session.run(translation['query'], translation['params'])
            records = list(result)
        
        results = []
        for record in records:
            row = {}
            for key in record.keys():
                value = record[key]
                if value is not None:
                    row[key] = str(value)
            if row:
                results.append(row)
        
        return jsonify({
            'answer': translation['answer'],
            'results': results,
            'cypher': translation['query'].strip(),
            'count': len(results),
            'intent': translation['intent'],
            'confidence': translation['confidence'],
            'entities': translation['entities']
        })
        
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """Get comprehensive graph statistics"""
    if not driver:
        return jsonify({'error': 'Not connected to Neo4j'}), 500
    
    try:
        with driver.session() as session:
            entity_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            rel_count = session.run("MATCH ()-[r]-() RETURN count(r) as count").single()['count']
            rel_types = session.run("MATCH ()-[r]-() RETURN count(DISTINCT type(r)) as count").single()['count']
            
            return jsonify({
                'total_entities': entity_count,
                'total_relationships': rel_count,
                'relationship_types': rel_types
            })
    except Exception as e:
        return jsonify({'error': f'Stats query failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting Advanced Knowledge Graph Query System...")
    print("📊 Open http://localhost:5000 in your browser")
    print("🧠 Advanced NLP enabled:", NLP_AVAILABLE)
    print("🔗 Neo4j connection:", NEO4J_URI)
    
    app.run(debug=True, host='0.0.0.0', port=5000)