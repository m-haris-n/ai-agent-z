import asyncio
import tiktoken
import re
import json

class Tokenizer:
    """Utility class for token-related operations."""
    
    @staticmethod
    def count_tokens(input_data, encoding_name: str = "cl100k_base") -> int:
        """
        Count the number of tokens in a given input (string or dictionary).
        
        Args:
            input_data (Union[str, dict]): Input text or dictionary to tokenize.
            encoding_name (str): Encoding to use for tokenization.
        
        Returns:
            int: Number of tokens.
        """
        try:
            # If input is a dictionary, convert it to a JSON string
            if isinstance(input_data, dict):
                input_data = json.dumps(input_data)
            elif not isinstance(input_data, str):
                raise ValueError("Input must be a string or dictionary")
            
            # Use tiktoken to count tokens
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(input_data))
        except Exception as e:
            print(f"Token counting error: {e}")
            return 0

def chunk_text(objects: list, max_tokens: int, tokenizer=None) -> list:
    """
    Divide objects into chunks respecting token limits and ensuring entire objects are included.

    Args:
        objects (list): List of input objects (strings) to chunk.
        max_tokens (int): Maximum tokens per chunk.
        tokenizer (Tokenizer, optional): Tokenizer to use.

    Returns:
        list: Chunks of objects.
    """
    if not tokenizer:
        tokenizer = Tokenizer()

    chunks = []
    current_chunk = []
    current_token_count = 0

    for obj in objects:
        
        obj_tokens = tokenizer.count_tokens(obj)

        # If adding this object would exceed max tokens, start a new chunk
        if current_token_count + obj_tokens > max_tokens:
            if current_chunk:  # Only add if there's something to add
                chunks.append(current_chunk)  # Ensure no extra spaces between objects
            current_chunk = [obj]
            current_token_count = obj_tokens
        else:
            current_chunk.append(obj)
            current_token_count += obj_tokens

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)  # Ensure no extra spaces between objects
    
    print(len(chunks))

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

def process_json(data: list) -> list:
    """
    Convert each ticket (dictionary) into a readable dictionary string format,
    handling nested dictionaries, and store each result in the tickets array.
    
    Args:
        data (list): A list of tickets, where each ticket is a dictionary.
    
    Returns:
        list: A list of strings, each representing a ticket in a readable dictionary format.
    """
    def dict_to_string(d):
        """
        Recursively converts a dictionary into a readable string format.
        """
        result = []
        for k, v in d.items():
            if isinstance(v, dict):  # If the value is a nested dictionary, process it recursively
                v = dict_to_string(v)
            else:
                v = str(v).replace(' ', '')  # Remove spaces from values
            result.append(f"'{k}': '{v}'")
        return '{' + ', '.join(result) + '}'
    
    tickets = []
    
    for ticket in data:
        # Convert the ticket into a readable dictionary-like string format
        ticket_str = dict_to_string(ticket)
        tickets.append(ticket_str)
    
    return tickets
