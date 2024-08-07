import requests
import json

# Your PaLM API key
api_key = 'AIzaSyDl1IdM-vqwgKKC6Z1dub1TcKwTzoQXl3Q'

# Endpoint for the PaLM API (replace with the correct URL)
url = 'https://your-palm-api-endpoint/v1/predict'  # Adjust as needed

# Define the prompt and model ID
prompt = "10 * 10"
model_id = 'your-model-id'  # Replace with the specific model ID

# Create the payload for the API request
payload = {
    'model': model_id,
    'inputs': {
        'text': prompt
    }
}

# Set up the headers with your API key
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

# Make the request to the PaLM API
response = requests.post(url, data=json.dumps(payload), headers=headers)

# Handle and print the response
if response.status_code == 200:
    result = response.json()
    print(result['predictions'][0]['text'])  # Adjust based on the response structure
else:
    print(f"Error: {response.status_code}")
    print(response.text)
