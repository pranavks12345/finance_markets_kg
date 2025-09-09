#!/usr/bin/env python3
"""
Accurate Training Pipeline - Scale Up
Generate high-quality training data from all your articles
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

load_dotenv()

class AccurateLLMNER:
    def __init__(self):
        """Your proven LLM system for training data generation"""
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.total_cost = 0.0
        
        self.schema = {
            "COMPANY": "Companies, organizations, financial institutions",
            "PERSON": "People mentioned (CEOs, analysts, politicians)",
            "MONEY": "Dollar amounts, currencies ($3.7B, €2M)",
            "PERCENTAGE": "Percentages (18%, 300%)",
            "FINANCIAL_TERM": "Financial concepts (bullish, dividend, earnings)",
            "TICKER": "Stock symbols (AAPL, MSFT, GOOGL)",
            "PRODUCT": "Specific products (iPhone, semiconductors)",
            "POLICY": "Policy terms (tariffs, sanctions)",
            "GEOPOL": "Countries, regions (United States, EU)"
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities using your proven prompt"""
        try:
            schema_desc = "\n".join([f"- {k}: {v}" for k, v in self.schema.items()])
            
            prompt = f"""Extract ALL financially relevant entities from this news text.

CATEGORIES:
{schema_desc}

RULES:
1. Extract EVERYTHING financially relevant
2. Include ALL people, companies, financial terms
3. Use canonical forms: "Apple Inc." not "Apple"
4. Be comprehensive but accurate

TEXT: "{text}"

Return JSON array:
[{{"text": "entity", "label": "CATEGORY"}}]

JSON:"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a financial entity extraction expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean response
            if content.startswith('```json'):
                content = content.replace('```json', '').replace('```', '').strip()
            if content.startswith('```'):
                content = content.replace('```', '').strip()
            
            entities = json.loads(content)
            
            # Estimate cost
            input_tokens = len(prompt) // 4
            output_tokens = len(content) // 4
            cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)
            self.total_cost += cost
            
            return self._validate_entities(entities)
            
        except Exception as e:
            print(f"Error extracting from text: {text[:50]}... - {e}")
            return []
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate extracted entities"""
        validated = []
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if not all(key in entity for key in ['text', 'label']):
                continue
            if entity['label'] not in self.schema:
                continue
            if len(entity['text'].strip()) < 2:
                continue
            
            validated.append({
                'text': entity['text'].strip(),
                'label': entity['label'],
                'confidence': 0.95
            })
        
        return validated

class AccurateTrainingDataGenerator:
    def __init__(self):
        """Generate high-quality training data"""
        self.llm_ner = AccurateLLMNER()
        
        # Complete label set for training
        self.labels = [
            'O',  # Outside
            'B-COMPANY', 'I-COMPANY',
            'B-PERSON', 'I-PERSON',
            'B-MONEY', 'I-MONEY',
            'B-PERCENTAGE', 'I-PERCENTAGE',
            'B-FINANCIAL_TERM', 'I-FINANCIAL_TERM',
            'B-TICKER', 'I-TICKER',
            'B-PRODUCT', 'I-PRODUCT',
            'B-POLICY', 'I-POLICY',
            'B-GEOPOL', 'I-GEOPOL'
        ]
        
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
    
    def convert_to_bio_labels(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        """Convert entities to BIO format with proper alignment"""
        # Tokenize text (simple whitespace splitting)
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        # Sort entities by position for consistent labeling
        entities = sorted(entities, key=lambda x: len(x['text']), reverse=True)
        
        for entity in entities:
            entity_text = entity['text']
            entity_label = entity['label']
            entity_tokens = entity_text.split()
            
            # Find the entity in the token list
            for i in range(len(tokens) - len(entity_tokens) + 1):
                # Check for exact match
                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                    # Apply BIO labeling only if not already labeled
                    if all(labels[i+j] == 'O' for j in range(len(entity_tokens))):
                        labels[i] = f'B-{entity_label}'
                        for j in range(1, len(entity_tokens)):
                            if i+j < len(labels):
                                labels[i+j] = f'I-{entity_label}'
                        break
                
                # Try case-insensitive match
                elif [t.lower() for t in tokens[i:i+len(entity_tokens)]] == [t.lower() for t in entity_tokens]:
                    if all(labels[i+j] == 'O' for j in range(len(entity_tokens))):
                        labels[i] = f'B-{entity_label}'
                        for j in range(1, len(entity_tokens)):
                            if i+j < len(labels):
                                labels[i+j] = f'I-{entity_label}'
                        break
        
        return tokens, labels
    
    def generate_training_data(self, articles: List[Dict], max_articles: int = 150) -> List[Dict]:
        """Generate training data from articles using LLM"""
        print(f"Generating training data from {min(len(articles), max_articles)} articles...")
        
        training_examples = []
        processed = 0
        
        for i, article in enumerate(articles[:max_articles]):
            if i % 10 == 0:
                print(f"Processing article {i+1}/{min(len(articles), max_articles)} (Cost: ${self.llm_ner.total_cost:.3f})")
            
            # Combine headline and summary
            text = article['headline']
            if article.get('summary') and len(article['summary'].strip()) > 10:
                text += ". " + article['summary']
            
            # Skip very long texts to control costs
            if len(text) > 1500:
                text = text[:1500]
            
            # Extract entities using LLM
            entities = self.llm_ner.extract_entities(text)
            
            if not entities:  # Skip articles with no entities
                continue
            
            # Convert to BIO format
            tokens, labels = self.convert_to_bio_labels(text, entities)
            
            if len(tokens) != len(labels):
                print(f"Token/label mismatch: {len(tokens)} vs {len(labels)}")
                continue
            
            training_examples.append({
                'text': text,
                'tokens': tokens,
                'labels': labels,
                'entities': entities,  # Keep for debugging
                'source': article.get('source', 'unknown')
            })
            
            processed += 1
        
        print(f"Generated {len(training_examples)} training examples")
        print(f"Total LLM cost: ${self.llm_ner.total_cost:.3f}")
        
        return training_examples

class AccurateNERTrainer:
    def __init__(self):
        """Trainer for high-accuracy local NER model"""
        self.model_name = "distilbert-base-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Labels from generator
        self.labels = [
            'O',  # Outside
            'B-COMPANY', 'I-COMPANY',
            'B-PERSON', 'I-PERSON',
            'B-MONEY', 'I-MONEY',
            'B-PERCENTAGE', 'I-PERCENTAGE',
            'B-FINANCIAL_TERM', 'I-FINANCIAL_TERM',
            'B-TICKER', 'I-TICKER',
            'B-PRODUCT', 'I-PRODUCT',
            'B-POLICY', 'I-POLICY',
            'B-GEOPOL', 'I-GEOPOL'
        ]
        
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.labels)}
    
    def tokenize_and_align_labels(self, examples):
        """Proper tokenization with label alignment"""
        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding=False,
            max_length=512
        )
        
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens
                elif word_idx != previous_word_idx:
                    # First subtoken of a word
                    label_ids.append(self.label_to_id[label[word_idx]])
                else:
                    # Continuation subtoken
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def prepare_datasets(self, training_examples: List[Dict]) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        # Convert to required format
        texts = [ex['tokens'] for ex in training_examples]
        labels = [ex['labels'] for ex in training_examples]
        
        # Split data
        if len(training_examples) < 10:
            train_texts, val_texts = texts, texts
            train_labels, val_labels = labels, labels
        else:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'tokens': train_texts,
            'labels': train_labels
        })
        
        val_dataset = Dataset.from_dict({
            'tokens': val_texts,
            'labels': val_labels
        })
        
        # Tokenize
        train_dataset = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        val_dataset = val_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset: Dataset, val_dataset: Dataset, output_dir: str = "./accurate-financial-ner"):
        """Train the accurate model"""
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels),
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=500,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return model

def load_all_articles():
    """Load all articles from your files"""
    articles = []
    data_dir = Path("data/raw/news")
    
    if not data_dir.exists():
        print("No data directory found")
        return []
    
    for json_file in data_dir.glob("*.json"):
        print(f"Loading {json_file.name}...")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:  # Take all articles, not just first 50
                    headline = item.get('headline', '') or item.get('title', '')
                    summary = item.get('summary', '') or item.get('description', '')
                    
                    if headline and len(headline.strip()) > 10:
                        articles.append({
                            'headline': headline.strip(),
                            'summary': summary.strip() if summary else '',
                            'source': json_file.name
                        })
                        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(articles)} total articles")
    return articles

def main():
    """Accurate training pipeline"""
    print("Accurate Financial NER Training Pipeline")
    print("=" * 50)
    
    # Load all articles
    articles = load_all_articles()
    
    if len(articles) < 10:
        print("Not enough articles for training")
        return
    
    # Generate training data (test with 10 articles first)
    generator = AccurateTrainingDataGenerator()
    training_examples = generator.generate_training_data(articles, max_articles=10)
    
    if len(training_examples) < 10:
        print("Not enough training examples generated")
        return
    
    # Save training data
    with open('training_data.json', 'w') as f:
        json.dump(training_examples, f, indent=2)
    print("Training data saved to training_data.json")
    
    # Train model
    trainer = AccurateNERTrainer()
    train_dataset, val_dataset = trainer.prepare_datasets(training_examples)
    model = trainer.train_model(train_dataset, val_dataset)
    
    print("Training complete!")
    print("Accurate local model ready for knowledge graph construction")

if __name__ == "__main__":
    main()