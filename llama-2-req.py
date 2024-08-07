import openai

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

# Loop through questions to create a conversation
for question in questions:
    # Add user question to conversation
    conversation.append({"role": "user", "content": question})
    
    # Get model's response
    response = openai.ChatCompletion.create(
        model="NousResearch/Llama-2-7b-chat-hf",
        messages=conversation
    )
    
    # Extract and print the model's response
    answer = response.choices[0].message["content"]
    print(f"Human: {question}")
    print(f"Assistant: {answer}\n")
    
    # Add model's response to conversation history
    conversation.append({"role": "assistant", "content": answer})
