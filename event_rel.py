#!/usr/bin/env python3
"""
Robust Financial News Relationship & Event Extraction Script
Bulletproof processing with rate limiting, retries, and checkpoints
"""

import json
import asyncio
import aiohttp
import ssl
import os
import subprocess
import signal
import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime, timedelta
import logging
import backoff

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# System prompt for extraction
SYSTEM_PROMPT = """You are an expert financial relationship and event extraction system. Your task is to analyze financial news articles and extract structured relationships and events that can be used to build a knowledge graph.

## Output Schema
Return a JSON object with:

{
  "relationships": [
    {
      "source_entity": "string",
      "relationship_type": "string", 
      "target_entity": "string",
      "confidence": 0.0-1.0,
      "evidence": "text from headline/summary",
      "metadata": {
        "financial_value": "string (if applicable)",
        "direction": "positive/negative/neutral",
        "temporal_indicator": "string (if applicable)"
      }
    }
  ],
  "events": [
    {
      "event_type": "string",
      "primary_entity": "string",
      "event_description": "string",
      "confidence": 0.0-1.0,
      "evidence": "text from headline/summary", 
      "metadata": {
        "financial_impact": "string (if applicable)",
        "market_direction": "positive/negative/neutral",
        "scale": "major/moderate/minor"
      }
    }
  ]
}

## Relationship Types
- MARKET_CAP_RANKING: Company A overtakes/surpasses Company B in valuation
- COMPETES_WITH: Companies in direct competition
- OUTPERFORMS: Company A performs better than Company B
- ACQUIRES: Company A acquires Company B
- PARTNERS_WITH: Strategic partnerships
- INVESTS_IN: Investment relationships
- EMPLOYED_BY: Person works at Company
- LEADS: Person is CEO/executive of Company
- ANALYST_COVERAGE: Analyst/firm covers stock
- PRICE_TARGET: Analyst sets target price
- RATING: Buy/Sell/Hold ratings
- REGULATES: Government entity regulates Company

## Event Types
- STOCK_MOVEMENT: Significant price changes, rallies, selloffs
- MARKET_CAP_CHANGE: Valuation milestones, ranking changes
- EARNINGS_RELEASE: Quarterly/annual results
- ACQUISITION: M&A activity
- PRODUCT_LAUNCH: New product/service announcements
- REGULATORY_ACTION: Government intervention, antitrust actions
- EARNINGS_BEAT: Exceeding expectations
- EXECUTIVE_CHANGE: Leadership appointments/departures

## Guidelines
1. Normalize entity names: "Microsoft Corp" → "Microsoft", "AAPL" → "Apple"
2. Confidence scoring: 0.9-1.0 (explicit), 0.7-0.9 (strong), 0.5-0.7 (reasonable)
3. Quote specific evidence from headline/summary
4. Extract financial values, percentages, dates when present
5. Only extract relationships/events with confidence >= 0.5

Return ONLY valid JSON, no additional text or explanations.

## Entity Mapping Guidelines
When extracting relationships, normalize entity names:
- "Apple Inc" or "AAPL" → "Apple"
- "Microsoft Corp" or "MSFT" → "Microsoft"  
- "Nvidia Corp" or "NVDA" → "Nvidia"
- "OpenAI Inc." → "OpenAI"
- Use company names, not tickers, for relationships
- Keep person names as provided
- Use official company names for consistency

## Important Notes
- Focus on the most significant relationships (confidence >= 0.7)
- Extract only relationships that are explicitly stated or strongly implied
- Use evidence quotes directly from headline/summary
- Ensure all JSON is properly formatted with no trailing commas"""

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    def __init__(self, requests_per_minute: int = 300, tokens_per_minute: int = 150000):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.request_times = []
        self.token_usage = []
        self.lock = threading.Lock()
        
    def can_make_request(self, estimated_tokens: int = 2000) -> Tuple[bool, float]:
        """Check if we can make a request and return wait time if not"""
        with self.lock:
            now = time.time()
            minute_ago = now - 60
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]
            
            # Check request limit
            if len(self.request_times) >= self.requests_per_minute:
                wait_time = 60 - (now - self.request_times[0])
                return False, wait_time
            
            # Check token limit
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
    """Robust checkpoint manager"""
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "extraction_checkpoint.json"
        self.results_file = self.output_dir / "extraction_partial_results.json"
        
    def save_checkpoint(self, successfully_processed: Set[int], failed_indices: Set[int], 
                       results: List[Dict], total_articles: int):
        """Save checkpoint with success/failure tracking"""
        checkpoint_data = {
            "successfully_processed": list(successfully_processed),
            "failed_indices": list(failed_indices),
            "results_count": len(results),
            "total_articles": total_articles,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save checkpoint metadata
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save actual results separately
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
            
            # Load partial results
            results = []
            if self.results_file.exists():
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
            
            return successfully_processed, failed_indices, results
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

def start_caffeinate():
    """Start caffeinate to prevent Mac from sleeping during processing"""
    try:
        caffeinate_process = subprocess.Popen(['caffeinate', '-d', '-i', '-s'], 
                                            stdout=subprocess.DEVNULL, 
                                            stderr=subprocess.DEVNULL)
        logger.info("✓ Caffeinate started - Mac will stay awake during processing")
        return caffeinate_process
    except FileNotFoundError:
        logger.warning("Caffeinate not found - running without sleep prevention")
        return None
    except Exception as e:
        logger.warning(f"Could not start caffeinate: {e}")
        return None

def stop_caffeinate(caffeinate_process):
    """Stop the caffeinate process"""
    if caffeinate_process:
        try:
            caffeinate_process.terminate()
            caffeinate_process.wait(timeout=5)
            logger.info("✓ Caffeinate stopped")
        except:
            try:
                caffeinate_process.kill()
            except:
                pass

class FinancialExtractor:
    def __init__(self, rate_limiter: RateLimiter):
        self.session = None
        self.rate_limiter = rate_limiter
        
    async def __aenter__(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3, base=2, max_value=60)
    async def extract_from_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relationships and events from a single article with rate limiting"""
        
        user_message = f"""
        Analyze this financial news article and extract relationships and events:

        **Headline:** {article['headline']}
        
        **Summary:** {article['summary']}
        
        **Entities:** {json.dumps(article['entities'], indent=2)}
        
        Extract all relevant relationships and events following the schema.
        """
        
        # Rate limiting
        estimated_tokens = len(user_message) // 4 + 2000  # Rough token estimate
        can_proceed, wait_time = self.rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            logger.info(f"Rate limit - waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
        
        try:
            response = await self._call_openai(user_message)
            
            # Record successful request
            tokens_used = len(response) // 3  # Rough estimate
            self.rate_limiter.record_request(tokens_used)
                
            # Parse JSON response
            response = response.strip()
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise Exception("No JSON found in response")
                
            json_str = response[json_start:json_end]
            extracted_data = json.loads(json_str)
            
            # Add article metadata
            extracted_data['article_id'] = article.get('id', hash(article['headline']))
            extracted_data['headline'] = article['headline']
            extracted_data['timestamp'] = datetime.now().isoformat()
            
            return extracted_data
            
        except Exception as e:
            logger.warning(f"Error processing article: {article['headline'][:50]}... - {str(e)}")
            raise  # Let backoff handle retries
    
    async def _call_openai(self, user_message: str) -> str:
        """Call OpenAI API with error handling"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 429:
                raise Exception("Rate limit exceeded")
            elif response.status != 200:
                raise Exception(f"API call failed: {response.status}")
            
            result = await response.json()
            return result["choices"][0]["message"]["content"]

async def process_single_article_with_retry(extractor: FinancialExtractor, article_idx: int, 
                                          article: Dict, max_retries: int = 5) -> Tuple[int, Optional[Dict], bool]:
    """Process single article with aggressive retry logic"""
    
    for attempt in range(max_retries + 1):
        try:
            result = await extractor.extract_from_article(article)
            return article_idx, result, True
            
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            
            if attempt < max_retries:
                if is_rate_limit:
                    wait_time = min(300, 30 * (2 ** attempt))  # 30s, 60s, 120s, 240s, 300s
                    logger.warning(f"Article {article_idx} rate limited (attempt {attempt + 1}), waiting {wait_time}s")
                else:
                    wait_time = min(60, 5 * (2 ** attempt))  # 5s, 10s, 20s, 40s, 60s
                    logger.warning(f"Article {article_idx} error (attempt {attempt + 1}): {e}, waiting {wait_time}s")
                
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Article {article_idx} failed after {max_retries + 1} attempts: {e}")
    
    # Create error entry for failed article
    error_result = {
        "relationships": [],
        "events": [],
        "article_id": article.get('id', hash(article['headline'])),
        "headline": article['headline'],
        "error": f"Failed after {max_retries + 1} attempts",
        "timestamp": datetime.now().isoformat()
    }
    
    return article_idx, error_result, False

async def process_batch_robust(extractor: FinancialExtractor, batch_articles: List[Tuple[int, Dict]], 
                             semaphore: asyncio.Semaphore) -> Tuple[List[Tuple[int, Dict]], Set[int]]:
    """Process batch with individual article retry logic"""
    
    async def process_with_semaphore(article_data):
        async with semaphore:
            article_idx, article = article_data
            return await process_single_article_with_retry(extractor, article_idx, article)
    
    # Process all articles in batch
    tasks = [process_with_semaphore(article_data) for article_data in batch_articles]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_results = []
    failed_indices = set()
    
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Unexpected batch error: {result}")
            failed_indices.update(idx for idx, _ in batch_articles)
        else:
            article_idx, article_result, success = result
            if success and not article_result.get('error'):
                successful_results.append((article_idx, article_result))
            else:
                successful_results.append((article_idx, article_result))  # Include error entries
                failed_indices.add(article_idx)
    
    return successful_results, failed_indices

def load_articles(file_path: str) -> List[Dict]:
    """Load articles from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: List[Dict], output_file: str):
    """Save extraction results to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def print_summary(results: List[Dict]):
    """Print extraction summary"""
    total_articles = len(results)
    total_relationships = sum(len(r.get('relationships', [])) for r in results)
    total_events = sum(len(r.get('events', [])) for r in results)
    errors = sum(1 for r in results if 'error' in r)
    
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"Articles processed: {total_articles}")
    print(f"Total relationships: {total_relationships}")
    print(f"Total events: {total_events}")
    print(f"Errors: {errors}")
    print(f"Success rate: {((total_articles-errors)/total_articles)*100:.1f}%")
    print(f"Average relationships per successful article: {total_relationships/max(total_articles-errors,1):.1f}")
    print(f"Average events per successful article: {total_events/max(total_articles-errors,1):.1f}")

async def robust_extraction_main():
    """Main extraction function with bulletproof error handling"""
    
    # Start caffeinate
    caffeinate_process = start_caffeinate()
    
    def signal_handler_with_caffeinate(sig, frame):
        logger.info("\n🛑 Stopping extraction...")
        if caffeinate_process:
            stop_caffeinate(caffeinate_process)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler_with_caffeinate)
    
    try:
        # Configuration
        INPUT_FILE = "/Users/pranavkumarsubha/finance_markets_ai/evaluation_results/cleaned_results_final_one.json"
        OUTPUT_FILE = "extracted_relationships_events_FULL.json"
        LIMIT = None  # Process ALL articles
        BATCH_SIZE = 5  # Conservative batch size
        MAX_CONCURRENT = 2  # Limit concurrent requests
        
        # Check API key
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY environment variable not set")
            return
        
        # Load articles
        logger.info(f"Loading articles from {INPUT_FILE}...")
        try:
            articles = load_articles(INPUT_FILE)
        except Exception as e:
            logger.error(f"Failed to load articles: {e}")
            return
        
        if LIMIT:
            articles = articles[:LIMIT]
        
        total_articles = len(articles)
        estimated_cost = total_articles * 0.005
        estimated_hours = total_articles * 12 / 3600  # 12 seconds per article
        
        logger.info(f"🚀 Processing {total_articles} articles using OpenAI GPT-4o-mini")
        logger.info(f"📊 Estimated cost: ${estimated_cost:.2f}")
        logger.info(f"⏱️  Estimated time: {estimated_hours:.1f} hours")
        
        # Setup rate limiter and checkpoint manager
        rate_limiter = RateLimiter(requests_per_minute=300, tokens_per_minute=150000)
        checkpoint_manager = CheckpointManager()
        
        # Load checkpoint if it exists
        successfully_processed = set()
        failed_indices = set()
        all_results = []
        
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            successfully_processed, failed_indices, all_results = checkpoint_data
            logger.info(f"Resuming: {len(successfully_processed)} successful, {len(failed_indices)} failed")
        
        # Identify articles that still need processing
        remaining_indices = []
        for i in range(total_articles):
            if i not in successfully_processed:
                remaining_indices.append(i)
        
        if not remaining_indices:
            logger.info("All articles already processed!")
            print_summary(all_results)
            return all_results
        
        logger.info(f"Processing {len(remaining_indices)} remaining articles")
        
        # Process articles in batches
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async with FinancialExtractor(rate_limiter) as extractor:
            for batch_start in range(0, len(remaining_indices), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(remaining_indices))
                batch_indices = remaining_indices[batch_start:batch_end]
                batch_articles = [(idx, articles[idx]) for idx in batch_indices]
                
                batch_num = (batch_start // BATCH_SIZE) + 1
                total_batches = (len(remaining_indices) - 1) // BATCH_SIZE + 1
                
                logger.info(f"Processing batch {batch_num}/{total_batches} (articles {batch_indices[0]+1}-{batch_indices[-1]+1})")
                
                # Process batch with robust error handling
                try:
                    batch_results, batch_failed = await process_batch_robust(extractor, batch_articles, semaphore)
                    
                    # Update tracking
                    for article_idx, result in batch_results:
                        if not result.get('error'):
                            successfully_processed.add(article_idx)
                        else:
                            failed_indices.add(article_idx)
                        all_results.append(result)
                    
                    logger.info(f"✓ Batch {batch_num} completed: {len(batch_results)} processed, {len(batch_failed)} failed")
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} completely failed: {e}")
                    # Mark all articles in batch as failed
                    for idx in batch_indices:
                        failed_indices.add(idx)
                        all_results.append({
                            "relationships": [],
                            "events": [],
                            "article_id": articles[idx].get('id', hash(articles[idx]['headline'])),
                            "headline": articles[idx]['headline'],
                            "error": f"Batch failure: {str(e)}",
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Save checkpoint every 10 batches
                if batch_num % 10 == 0:
                    checkpoint_manager.save_checkpoint(successfully_processed, failed_indices, all_results, total_articles)
                
                # Progress update
                completed = len(successfully_processed)
                current_cost = completed * 0.005
                progress_pct = (completed / total_articles) * 100
                
                if batch_num % 5 == 0:  # Update every 5 batches
                    logger.info(f"Progress: {completed}/{total_articles} ({progress_pct:.1f}%) - Cost: ${current_cost:.2f}")
        
        # Final retry for failed articles
        if failed_indices and len(failed_indices) < 100:  # Only retry if manageable number
            logger.info(f"Final retry for {len(failed_indices)} failed articles")
            
            retry_articles = [(idx, articles[idx]) for idx in failed_indices if idx < len(articles)]
            
            async with FinancialExtractor(rate_limiter) as retry_extractor:
                for article_idx, article in retry_articles:
                    try:
                        result = await retry_extractor.extract_from_article(article)
                        # Update the failed result in all_results
                        for i, existing_result in enumerate(all_results):
                            if existing_result.get('article_id') == result.get('article_id'):
                                all_results[i] = result
                                successfully_processed.add(article_idx)
                                failed_indices.discard(article_idx)
                                logger.info(f"Successfully retried article {article_idx}")
                                break
                        
                        await asyncio.sleep(3)  # Conservative delay for retries
                        
                    except Exception as e:
                        logger.warning(f"Article {article_idx} still failing on final retry: {e}")
        
        # Save final results
        save_results(all_results, OUTPUT_FILE)
        logger.info(f"✅ Results saved to {OUTPUT_FILE}")
        
        # Final summary
        print_summary(all_results)
        
        success_rate = len(successfully_processed) / total_articles
        final_cost = len(successfully_processed) * 0.005
        
        logger.info(f"🎉 EXTRACTION COMPLETE!")
        logger.info(f"Success rate: {success_rate:.1%}")
        logger.info(f"Final cost: ${final_cost:.2f}")
        
        return all_results
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
        
    finally:
        if caffeinate_process:
            stop_caffeinate(caffeinate_process)

async def main():
    """Main entry point with comprehensive error handling"""
    try:
        results = await robust_extraction_main()
        return results
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.info("Check logs for details. You can resume by running the script again.")

if __name__ == "__main__":
    asyncio.run(main())