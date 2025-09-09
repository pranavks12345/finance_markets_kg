# finance_markets_kg
Natural language interface for querying financial knowledge graphs built from Finnhub news data. Ask questions in plain English about companies, relationships, and market events.
Features

Natural language processing for complex questions
Smart query translation to Cypher
Real-time results with confidence scores
Web interface with sample questions
Built from real financial news data (AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA)

## Quick Start
pipenv install


##Key Files

frontend.py - Main web interface for querying //
kg_loader_neo4j.py - Loads processed data into Neo4j //
test_api.py - Pulls news from Finnhub API //
preprocessing_kg.py - Cleans and prepares raw data //
ner_for_all.py - Extracts entities from text
event_rel.py - Extracts relationships and events
Pipfile - Python dependencies
fine_tuning_model.py - Fine tunes FinBert Model
