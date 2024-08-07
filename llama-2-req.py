import openai
import requests
import json

# Set OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"  # Not needed since we're using a local server
openai.api_base = "http://localhost:8000/v1"

questions = [
    "What is the significance and history of Congo Square?",
    "Does it still exist?",
    "How did it influence jazz music?",
    "tell a few stories about Congo Square",
]

# Initialize conversation with system message
conversation = [{"role": "system", "content": "You are a helpful assistant."}]

# Function to handle streaming responses
def handle_streaming_response(response):
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data = decoded_line[len("data: "):]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                yield chunk

# Loop through questions to create a conversation
for question in questions:
    # Add user question to conversation
    conversation.append({"role": "user", "content": question})

    # Send the request and get the streaming response
    response = requests.post(
        f"{openai.api_base}/chat/completions",
        headers={"Authorization": f"Bearer {openai.api_key}"},
        json={
            "model": "NousResearch/Llama-2-7b-chat-hf",
            "messages": conversation,
            "stream": True  # Enable streaming
        },
        stream=True
    )
    
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        continue
    
    # Process the streaming response
    full_response = ""
    for chunk in handle_streaming_response(response):
        content = chunk['choices'][0]['delta'].get('content', '')
        print(content, end='', flush=True)
        full_response += content
    print()  # Print a newline at the end of the answer

    # Add model's response to conversation history
    conversation.append({"role": "assistant", "content": full_response})
