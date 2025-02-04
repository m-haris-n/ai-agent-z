from fastapi import FastAPI, HTTPException, Depends
from aiolimiter import AsyncLimiter
from pydantic import BaseModel
import asyncio
from typing import Dict, Any
import json
from .config import Config
from .processing import AIProcessor
from .utils import Tokenizer
from datetime import datetime, timedelta
from functools import wraps

# Validate config on startup
Config.validate()

app = FastAPI(
    title="AI Data Processor",
    description="Process large JSON data using Claude via Anthropic API"
)

class ProcessRequest(BaseModel):
    data: Dict[str, Any]
    user_prefix: str

class ProcessResponse(BaseModel):
    result: str

processor = AIProcessor()

# Define rate limiter: 10 requests per minute
rate_limiter = AsyncLimiter(10, 60)

# Token limit configurations
MAX_INPUT_TOKENS = 60000
MAX_OUTPUT_TOKENS = 4096
GLOBAL_TOKEN_LIMIT = 1_000_000  # Global token limit for all users combined

# Global token tracker
global_token_usage = {
    "total_tokens": 0,
    "reset_time": datetime.utcnow() + timedelta(minutes=60)
}

# Periodic token tracker reset
async def reset_global_tokens():
    """
    Resets the global token tracker periodically.
    """
    while True:
        now = datetime.utcnow()
        if now >= global_token_usage["reset_time"]:
            global_token_usage["total_tokens"] = 0
            global_token_usage["reset_time"] = now + timedelta(minutes=1)
        await asyncio.sleep(1)

# Retry decorator
def retry_on_limit_reached(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except HTTPException as e:
                    if "limit" in str(e.detail).lower() and retries < max_retries - 1:
                        retries += 1
                        await asyncio.sleep(delay)  # Wait before retrying
                    else:
                        raise
        return wrapper
    return decorator

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(reset_global_tokens())

# Dependency to enforce rate limiting and global token tracking
async def limit_requests():
    async with rate_limiter:
        pass

# Dependency to enforce global token limits
async def validate_global_tokens(request: ProcessRequest):
    """
    Validates the global token usage and updates it for each request.
    """
    try:
        # Count tokens in request data
        data_token_count = Tokenizer.count_tokens(request.data)
        user_prefix_token_count = Tokenizer.count_tokens(request.user_prefix)
        total_request_tokens = data_token_count + user_prefix_token_count

        # Validate against global token limit
        if global_token_usage["total_tokens"] + total_request_tokens > GLOBAL_TOKEN_LIMIT:
            raise HTTPException(
                status_code=429, detail="Global token limit reached. Please try again later."
            )

        # Update global token tracker
        global_token_usage["total_tokens"] += total_request_tokens
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    return request

@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(limit_requests)])
@retry_on_limit_reached(max_retries=3, delay=2)
async def process_data(request: ProcessRequest, validated_request=Depends(validate_global_tokens)):
    """
    Single endpoint to process input data using AI.
    """
    try:
        # Process the request
        result = await processor.process_data(
            data=request.data,
            user_prefix=request.user_prefix,
        )

        # Count tokens in the result
        output_token_count = Tokenizer.count_tokens(result)

        # Validate the output token count
        if output_token_count > MAX_OUTPUT_TOKENS:
            raise HTTPException(
                status_code=500,
                detail=f"Output exceeds token limit of {MAX_OUTPUT_TOKENS} tokens."
            )

        # Update global token tracker for output tokens
        global_token_usage["total_tokens"] += output_token_count

        return ProcessResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
