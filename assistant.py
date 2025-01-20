import os
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

def chat():
    # Initialize conversation history
    conversation_history = []
    
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
        
        # Get AI response
        response = llm.invoke(conversation_history)
        
        # Add AI response to history
        conversation_history.append(response)
        
        # Print AI response
        print("Bot:", response.content)

if __name__ == "__main__":
    chat()
