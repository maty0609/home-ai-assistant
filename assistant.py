import os
import psycopg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_postgres import PostgresChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
    streaming=True,
)

conn_info = os.getenv('DB_POSTGRES_URL')
sync_connection = psycopg.connect(conn_info)

# Create a session ID - static for now but in the future this will be for switching between different chats
session_id = "703801c1-6132-414a-8636-a947d967349e"
# Define table name
table_name = "chat_history"

def print_chat_history(messages):
    print("\n=== Previous Conversation ===")
    for message in messages:
        if isinstance(message, SystemMessage):
            continue  # Skip system messages
        prefix = "Bot: " if isinstance(message, AIMessage) else "You: "
        print(f"{prefix}{message.content}\n")
    print("======================\n")

def chat():
    system_prompt = """You are a helpful AI assistant. 
    - Be concise and clear in your responses
    - If you're not sure about something, admit it
    - Be friendly and professional
    - When discussing code, provide examples when helpful
    - Format your responses using markdown"""
    
    # Initialize chat history
    chat_message_history = PostgresChatMessageHistory("chat_history",session_id,sync_connection=sync_connection)
    
    conversation_messages = []
    # Add system prompt if history is empty
    if not chat_message_history.messages:
        chat_message_history.add_message(SystemMessage(content=system_prompt))
    
    
    conversation_messages = [SystemMessage(content=system_prompt)] + chat_message_history.messages
    # Print existing chat history
    if len(chat_message_history.messages) > 1:  # If there's more than just the system prompt  
        print_chat_history(chat_message_history.messages)

    print("Bot: Hello! I'm your AI assistant. Type 'quit' to end the conversation.")
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for quit command
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye!")
            break
                
        # Add user message to both history and current context
        user_message = HumanMessage(content=user_input)
        chat_message_history.add_message(user_message)
        conversation_messages.append(user_message)

        # Get AI response with streaming
        print("Bot: ", end="", flush=True)
        response_content = ""
        for chunk in llm.stream(conversation_messages):
            content_chunk = chunk.content
            print(content_chunk, end="", flush=True)
            response_content += content_chunk
        print()  # New line after response
        
        # Add AI response to both history and current context
        ai_message = AIMessage(content=response_content)
        chat_message_history.add_message(ai_message)
        conversation_messages.append(ai_message)
    
if __name__ == "__main__":
    chat()
