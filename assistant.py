import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
    streaming=True,  # Enable streaming
)

def chat():
    # Initialize conversation history with system prompt
    conversation_history = [
        SystemMessage(content="""You are a helpful AI assistant. 
        - Be concise and clear in your responses
        - If you're not sure about something, admit it
        - Be friendly and professional
        - When discussing code, provide examples when helpful
        - Format your responses using markdown""")
    ]
    
    print("Bot: Hello! I'm your AI assistant. Type 'quit' to end the conversation.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
        
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Get AI response with streaming
        print("Bot: ", end="", flush=True)
        response_content = ""
        for chunk in llm.stream(conversation_history):
            content_chunk = chunk.content
            print(content_chunk, end="", flush=True)
            response_content += content_chunk
        print()  # New line after response
        
        # Add AI response to history
        conversation_history.append(HumanMessage(content=response_content))

if __name__ == "__main__":
    chat()
