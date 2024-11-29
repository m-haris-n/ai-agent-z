# AI Data Processing Agent

## Project Setup

### Prerequisites
- Python 3.8+
- Anthropic API Key

### Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Configuration
1. Copy `.env.example` to `.env`
2. Add your Anthropic API key to `.env`

### Running the Server
```bash
uvicorn src.main:app --reload
```

### Usage Example
```python
import requests

response = requests.post('http://localhost:8000/process', json={
    'data': {'your': 'input_data'},
    'user_prefix': 'Your processing instructions'
})
print(response.json())
```

## Notes
- Ensure you have a valid Anthropic API key
- Customize `process_data` method in `processing.py` for specific use cases