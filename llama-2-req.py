import openai

# Set OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"  # Not needed since we're using a local server
openai.api_base = "http://localhost:8000/v1"

response = openai.ChatCompletion.create(
    model="/opt/data/models/Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Congo Square's significance and history ?"},
    ]
)

print(response.choices[0].message["content"])
