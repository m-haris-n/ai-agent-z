import json
import requests


# Loading Ticket Data
def process_tickets(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

file_path = r'C:\Users\HP\Desktop\Zendesk-Llm\res.json'

# Process the tickets
data = process_tickets(file_path)

response = requests.post('http://localhost:8000/process', json={
    'data': data,
    'user_prefix': 'Give me a summary of the data'
})
print(response.json())