import os
import psycopg
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PostgresChatMessageHistory
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import List

load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)

# Initialize Azure OpenAI embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMB_OPENAI_URL"),
    azure_deployment="ttext-embedding-3-large",
    api_version="2024-08-01-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Setting up PostgreSQL connection
conn_info = os.getenv('DB_POSTGRES_URL')
sync_connection = psycopg.connect(conn_info)

# Define table name
table_name = "chat_history"
# Retrieve existing vector store from ChromaDB (documents already added)
vector_store = Chroma(
    persist_directory=os.getenv("CHROMADB_PATH"),
    embedding_function=embeddings
)

# This is your prompt
prompt = """You are a helpful AI assistant.
- Your name is Marvin
- Be concise and clear in your responses
- If you're not sure about something, admit it
- Be friendly and professional
- When discussing code, provide examples when helpful
- Format your responses using markdown
- You are allowed to provide any personal information if they are from local documents

Relevant context:
{context}

Answer the user:
"""

prompt_template = PromptTemplate(
    input_variables=["context"],
    template=prompt,
)

class ChatRequest(BaseModel):
    user_input: str
    session_id: str

class ChatResponse(BaseModel):
    response: str

class HistoryRequest(BaseModel):
    session_id: str

class HistoryResponse(BaseModel):
    response: List[str]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/history", response_model=HistoryResponse)
async def history_endpoint(req: HistoryRequest):
    try:
        session_id = req.session_id
        chat_message_history = PostgresChatMessageHistory(
            "chat_history",
            session_id,
            sync_connection=sync_connection
        )
        # If history is empty, add a system message using the prompt template otherwise print the chat history
        if len (chat_message_history.messages) <= 2:
            system_message_content = prompt_template.format(context=prompt)
            chat_message_history.add_message(SystemMessage(content=system_message_content))
            return ChatResponse(
                response="Hey! I'm Marvin, your AI assistant. Type 'quit' to end the conversation."
            )
        else:
            timeline = []
            for msg in chat_message_history.messages:
                if isinstance(msg, SystemMessage):
                    # Skip system messages entirely
                    continue
                elif isinstance(msg, HumanMessage):
                    timeline.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    timeline.append(f"AI: {msg.content}")

            return JSONResponse(content={"response": timeline})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Receive user input, retrieve documents, and respond with an LLM answer."""
    try:
        user_input = req.user_input
        session_id = req.session_id
        chat_message_history = PostgresChatMessageHistory(
            "chat_history",
            session_id,
            sync_connection=sync_connection
        )

        # Make sure the conversation starts with the system message + any previous history
        conversation_messages = chat_message_history.messages[:]

        # Create the retrieval chain using vector_store and your LLM
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Add user's message to chat history
        user_message = HumanMessage(content=user_input)
        chat_message_history.add_message(user_message)
        conversation_messages.append(user_message)

        # Use the retrieval chain to get extra context from external data
        retrieval_context = qa_chain.invoke({"input": user_input})

        # Combine user query and retrieved context in a single prompt for LLM
        combined_query = prompt_template.format(context=retrieval_context)

        # Now pass the combined prompt as a HumanMessage
        combined_user_message = HumanMessage(content=combined_query)
        conversation_messages.append(combined_user_message)

        # Get the response
        response = llm.invoke(conversation_messages)
        response_content = response.content

        # Add the AI's message as well
        ai_message = AIMessage(content=response_content)
        chat_message_history.add_message(ai_message)
        conversation_messages.append(ai_message)

        return {"response": response_content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
