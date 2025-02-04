import asyncio
import anthropic
from anthropic import AsyncAnthropic
from aiolimiter import AsyncLimiter
from typing import Dict, Any, List
from .config import Config
from .utils import Tokenizer, chunk_text, gather_with_concurrency, process_json
from .decorators import retry, LlmCallError

class AIProcessor:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.tokenizer = Tokenizer()
    
    @retry(times=3, exceptions=(LlmCallError))
    async def process_chunk(self, chunk: str, user_prefix: str) -> str:
        """
        Process a single chunk of data using Claude.
        
        Args:
            chunk (str): Data chunk to process
            user_prefix (str): Prefix instruction for processing
        
        Returns:
            str: Processed result
        """
        try:
            response = await self.client.messages.create(
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "user", "content": f"{user_prefix}\n\nData: {chunk}"}
                ]
            )
            return response.content[0].text
        except Exception as e:
            raise LlmCallError("LLm Call Failed: ", 503)
    
    async def aggregate_results(self, results: List[str], user_prefix: str) -> str:
        """
        Aggregate results from multiple chunks, handling large result sets recursively.
        
        Args:
            results (List[str]): Results to aggregate
            user_prefix (str): Aggregation instruction
        
        Returns:
            str: Final aggregated result
        """
        print(f"Aggregating {len(results)} results")
        # If results fit in context, aggregate directly
        combined_results = "\n\n".join(results)
        if self.tokenizer.count_tokens(combined_results) <= Config.MAX_OUTPUT_TOKENS:
            response = await self.client.messages.create(
                model=Config.MODEL_NAME,
                max_tokens=Config.MAX_OUTPUT_TOKENS,
                messages=[
                    {"role": "user", "content": f"{user_prefix}\n\nResults to aggregate:\n{combined_results}"}
                ]
            )
            return response.content[0].text
        
        # If too large, recursively split and aggregate
        mid = len(results) // 2
        first_half = await self.aggregate_results(results[:mid], user_prefix)
        second_half = await self.aggregate_results(results[mid:], user_prefix)
        
        return await self.aggregate_results([first_half, second_half], user_prefix)
    
    async def process_data(self, data: Dict[str, Any], user_prefix: str, max_concurrency: int = 5) -> str:
        """
        Main data processing method.
        
        Args:
            data (Dict): Input JSON data
            user_prefix (str): Processing instructions
            max_concurrency (int): Max concurrent API calls
        
        Returns:
            str: Final processed result
        """
        # Convert data to processable string (customize as needed)
        data_str = process_json(data['tickets']) if data and 'tickets' in data else ''

        # Chunk the data
        chunks = chunk_text(data_str, Config.MAX_CONTEXT_TOKENS - self.tokenizer.count_tokens(user_prefix))

        # Process chunks in batches of max_concurrency
        results = []
        for i in range(0, len(chunks), max_concurrency):
            # Get the current batch of chunks
            current_batch = chunks[i:i + max_concurrency]   

            # Create tasks for the current batch
            chunk_tasks = [self.process_chunk(chunk, user_prefix) for chunk in current_batch]
            
            # Wait for all tasks in the batch to complete
            batch_results = await gather_with_concurrency(max_concurrency, *chunk_tasks)
            results.extend(batch_results)
        

        # Aggregate results
        final_result = await self.aggregate_results(
            results,
            "Aggregate these results coherently. If there is nothing to aggregate, return the original data without saying anything extra. Don't say 'Here is the aggregated result:' or anything like that."
        )
        
        return final_result