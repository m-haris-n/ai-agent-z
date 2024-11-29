from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from typing import Dict, Any

from .config import Config
from .processing import AIProcessor

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

@app.post("/process", response_model=ProcessResponse)
async def process_data(request: ProcessRequest):
    """
    Single endpoint to process input data using AI.
    
    Args:
        request (ProcessRequest): Input data and processing instructions
    
    Returns:
        ProcessResponse: Processed result
    """
    try:
        result = await processor.process_data(
            data=request.data, 
            user_prefix=request.user_prefix
        )
        return ProcessResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)