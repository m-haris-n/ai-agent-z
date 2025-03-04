import asyncio
import tiktoken

class Tokenizer:
    """Utility class for token-related operations."""
    
    @staticmethod
    def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count the number of tokens in a given text.
        
        Args:
            text (str): Input text to tokenize
            encoding_name (str): Encoding to use for tokenization
        
        Returns:
            int: Number of tokens
        """
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Token counting error: {e}")
            return 0

def chunk_text(text: str, max_tokens: int, tokenizer=None) -> list:
    """
    Divide text into chunks respecting token limits.
    
    Args:
        text (str): Input text to chunk
        max_tokens (int): Maximum tokens per chunk
        tokenizer (Tokenizer, optional): Tokenizer to use
    
    Returns:
        list: Chunks of text
    """
    if not tokenizer:
        tokenizer = Tokenizer()
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Split text into words or logical segments
    segments = text.split()
    
    for segment in segments:
        segment_tokens = tokenizer.count_tokens(segment)
        
        # If adding this segment would exceed max tokens, start a new chunk
        if current_token_count + segment_tokens > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        current_chunk.append(segment)
        current_token_count += segment_tokens
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

async def gather_with_concurrency(n: int, *tasks):
    """
    Run tasks with limited concurrency.
    
    Args:
        n (int): Number of concurrent tasks
        *tasks: Async tasks to run
    
    Returns:
        list: Results of tasks
    """
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(*(sem_task(task) for task in tasks))