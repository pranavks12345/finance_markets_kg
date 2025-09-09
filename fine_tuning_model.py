

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support

from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

@dataclass
class NERExample:
    tokens: List[str]
    labels: List[str]
    text: str

CONFIG = {
    "model_name": "ProsusAI/finbert",
    "output_dir": "/content/best-finbert-ner",
    "dataset_file": "/content/production_gold_dataset_5k.json",
    "epochs": 8,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "max_length": 512
}

print("Configuration loaded:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

class FinBERTDataProcessor:
    """Optimized data processor for Colab environment"""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.label_list = [
            "O",
            "B-COMPANY", "I-COMPANY",
            "B-PERSON", "I-PERSON", 
            "B-TICKER", "I-TICKER",
            "B-MONEY", "I-MONEY",
            "B-PERCENTAGE", "I-PERCENTAGE",
            "B-TIME_PERIOD", "I-TIME_PERIOD",
            "B-LOCATION", "I-LOCATION",
            "B-PRODUCT", "I-PRODUCT",
            "B-INDEX", "I-INDEX",
            "B-GOV_ORG", "I-GOV_ORG",
            "B-CONCEPT", "I-CONCEPT",
            "B-POLICY", "I-POLICY",
            "B-COUNTRY", "I-COUNTRY",
            "B-EVENT", "I-EVENT",
            "B-ALIAS", "I-ALIAS",
            "B-PUBLISHER", "I-PUBLISHER"
        ]
        
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
        
        print(f"Loaded {len(self.label_list)} labels")
        
    def load_dataset(self, json_file: str) -> List[NERExample]:
        """Load dataset from JSON file"""
        print(f"Loading dataset from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
            print("Please upload your dataset file to Colab first!")
            return []
        
        examples = []
        processed_count = 0
        
        for article in raw_data:
            text = f"{article.get('headline', '')} {article.get('summary', '')}"
            entities = article.get('entities', [])
            
            if not entities or len(text.strip()) < 10:
                continue
                
            ner_example = self._create_ner_example(text, entities)
            if ner_example:
                examples.append(ner_example)
                processed_count += 1
        
        print(f"✅ Processed {processed_count} articles into {len(examples)} training examples")
        return examples
    
    def _create_ner_example(self, text: str, entities: List[Dict]) -> Optional[NERExample]:
        """Convert entities to BIO format"""
        char_labels = ['O'] * len(text)
        
        for entity in entities:
            entity_text = entity.get('text', '').strip()
            entity_label = entity.get('label', '').upper()
            
            if not entity_text or entity_label not in [label.split('-')[1] for label in self.label_list if label != 'O']:
                continue
            
            pos = text.find(entity_text)
            if pos != -1:
                for i, char_pos in enumerate(range(pos, pos + len(entity_text))):
                    if char_pos < len(char_labels):
                        if i == 0:
                            char_labels[char_pos] = f"B-{entity_label}"
                        else:
                            char_labels[char_pos] = f"I-{entity_label}"
        
        words = text.split()
        word_labels = []
        char_pos = 0
        
        for word in words:
            word_start = text.find(word, char_pos)
            if word_start >= 0 and word_start < len(char_labels):
                word_labels.append(char_labels[word_start])
            else:
                word_labels.append('O')
            char_pos = word_start + len(word) if word_start >= 0 else char_pos + 1
        
        if len(words) != len(word_labels):
            return None
            
        return NERExample(tokens=words, labels=word_labels, text=text)
    
    def create_datasets(self, examples: List[NERExample]) -> DatasetDict:
        """Create train/val/test datasets"""
        print(f"Creating datasets from {len(examples)} examples")
        
        train_examples, temp_examples = train_test_split(
            examples, test_size=0.25, random_state=42, shuffle=True
        )
        val_examples, test_examples = train_test_split(
            temp_examples, test_size=0.4, random_state=42, shuffle=True
        )
        
        print(f"Dataset splits:")
        print(f"  Train: {len(train_examples)}")
        print(f"  Validation: {len(val_examples)}")
        print(f"  Test: {len(test_examples)}")
        
        return DatasetDict({
            'train': self._examples_to_dataset(train_examples),
            'validation': self._examples_to_dataset(val_examples),
            'test': self._examples_to_dataset(test_examples)
        })
    
    def _examples_to_dataset(self, examples: List[NERExample]) -> Dataset:
        """Convert to HuggingFace Dataset"""
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for example in examples:
            tokenized = self.tokenizer(
                example.tokens,
                is_split_into_words=True,
                truncation=True,
                padding=False,
                max_length=CONFIG["max_length"]
            )
            
            word_ids = tokenized.word_ids()
            aligned_labels = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != previous_word_idx:
                    if word_idx < len(example.labels):
                        aligned_labels.append(self.label2id.get(example.labels[word_idx], 0))
                    else:
                        aligned_labels.append(0)
                else:
                    if word_idx < len(example.labels):
                        label = example.labels[word_idx]
                        if label.startswith("B-"):
                            i_label = label.replace("B-", "I-")
                            aligned_labels.append(self.label2id.get(i_label, 0))
                        else:
                            aligned_labels.append(self.label2id.get(label, 0))
                    else:
                        aligned_labels.append(0)
                previous_word_idx = word_idx
            
            all_input_ids.append(tokenized['input_ids'])
            all_attention_masks.append(tokenized['attention_mask'])
            all_labels.append(aligned_labels)
        
        return Dataset.from_dict({
            'input_ids': all_input_ids,
            'attention_mask': all_attention_masks,
            'labels': all_labels
        })

from google.colab import files
import os

print("Upload your dataset JSON file:")
uploaded = files.upload()

dataset_filename = list(uploaded.keys())[0]
print(f"Using uploaded file: {dataset_filename}")

processor = FinBERTDataProcessor(CONFIG["model_name"])
examples = processor.load_dataset(dataset_filename)

if len(examples) > 0:
    datasets = processor.create_datasets(examples)
    print("Dataset loaded successfully!")
else:
    raise ValueError("No valid examples found in dataset")

def compute_metrics(eval_pred):
    """Compute F1, precision, recall"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [processor.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [processor.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    flat_true = [item for sublist in true_labels for item in sublist]
    flat_pred = [item for sublist in true_predictions for item in sublist]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, average='weighted', zero_division=0
    )
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

print("🤖 Loading FinBERT model...")
model = AutoModelForTokenClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=len(processor.label_list),
    id2label=processor.id2label,
    label2id=processor.label2id,
    ignore_mismatched_sizes=True
)

print("🔒 Freezing early layers...")
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
for layer in model.bert.encoder.layer[:6]:
    for param in layer.parameters():
        param.requires_grad = False

print("✅ Model setup complete!")

output_dir = Path(CONFIG["output_dir"])
output_dir.mkdir(exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=2,
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_ratio=0.1,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    remove_unused_columns=False,
    fp16=True,
    dataloader_pin_memory=False,
    report_to=None,
    seed=42
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=processor.tokenizer,
    padding=True,
    max_length=CONFIG["max_length"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("🏃‍♂️ Trainer ready!")

print("🚀 Starting training...")
print(f"Training on {len(datasets['train'])} examples")
print(f"Validating on {len(datasets['validation'])} examples")

trainer.train()

print("✅ Training completed!")

print("📊 Evaluating on test set...")
test_results = trainer.evaluate(datasets["test"])
print("Test Results:")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

print("💾 Saving model...")
trainer.save_model()
processor.tokenizer.save_pretrained(output_dir)

training_info = {
    "model_name": CONFIG["model_name"],
    "num_train_examples": len(datasets["train"]),
    "num_val_examples": len(datasets["validation"]),
    "num_test_examples": len(datasets["test"]),
    "num_labels": len(processor.label_list),
    "labels": processor.label_list,
    "test_results": test_results
}

with open(output_dir / "training_info.json", "w") as f:
    json.dump(training_info, f, indent=2)

print(f"✅ Model saved to {output_dir}")
print("🎉 Training pipeline complete!")

print("📦 Creating model archive for download...")
import shutil
shutil.make_archive('/content/finbert-ner-model', 'zip', CONFIG["output_dir"])
print("✅ Model archived as: /content/finbert-ner-model.zip")
print("You can download this file from the Files panel")

def test_model(text_sample):
    """Quick test function"""
    inputs = processor.tokenizer(text_sample, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    tokens = processor.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [processor.id2label[pred.item()] for pred in predictions[0]]
    
    print("Sample prediction:")
    for token, label in zip(tokens[:20], labels[:20]):
        if token.startswith("##"):
            continue
        print(f"  {token:15} -> {label}")

sample_text = "Apple Inc. reported revenue of $89.5 billion, up 15% from last quarter."
print(f"Testing with: {sample_text}")
test_model(sample_text)