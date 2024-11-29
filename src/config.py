import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    MODEL_NAME = os.getenv('MODEL_NAME', 'claude-3-5-sonnet-20240620')
    MAX_CONTEXT_TOKENS = int(os.getenv('MAX_CONTEXT_TOKENS', 200000))
    MAX_OUTPUT_TOKENS = int(os.getenv('MAX_OUTPUT_TOKENS', 4096))

    @classmethod
    def validate(cls):
        """Validate critical configuration parameters."""
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY must be set in .env file")