import os
import psycopg
import uvicorn
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PostgresChatMessageHistory
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
#from llama_index.core.retrievers import VectorStoreRetriever
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import chromadb
from langchain.prompts import PromptTemplate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.indices.vector_store.retrievers import VectorIndexAutoRetriever

load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)

# Initialize Azure OpenAI embeddings for LangChain (kept for compatibility)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMB_OPENAI_URL"),
    azure_deployment="ttext-embedding-3-large",
    api_version="2024-08-01-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize LlamaIndex Azure OpenAI embeddings
llama_embeddings = AzureOpenAIEmbedding(
    azure_deployment_name="ttext-embedding-3-large",
    azure_endpoint=os.getenv("EMB_OPENAI_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_version="2024-08-01-preview",
)

# Set LlamaIndex global settings
Settings.embed_model = llama_embeddings
Settings.llm = llm

# Setting up PostgreSQL connection
conn_info = os.getenv('DB_POSTGRES_URL')
sync_connection = psycopg.connect(conn_info)

# Define table name
table_name = "chat_history"

# Setup ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PATH"))
#chroma_collection = chroma_client.get_or_create_collection("documents")
chroma_collection = chroma_client.get_or_create_collection("example_collection")
# Create LlamaIndex vector store
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index from the existing vector store
index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context
)

# Create query engine directly from the index
# This is the simpler approach that should work reliably
query_engine = index.as_query_engine(similarity_top_k=3)

# Alternatively, if you specifically need VectorIndexAutoRetriever:
# from llama_index.core.vector_stores import VectorStoreInfo
# vector_store_info = VectorStoreInfo(
#     content_info="Document collection",
#     metadata_info=[]
# )
# auto_retriever = VectorIndexAutoRetriever(index, vector_store_info=vector_store_info)
# query_engine = RetrieverQueryEngine(auto_retriever)

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
            return JSONResponse(
                content={"response": ["Hey! I'm Marvin, your AI assistant."]}
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

@app.post("/chat")
async def stream_chat(chat_request: ChatRequest) -> StreamingResponse:
    user_input = chat_request.user_input
    session_id = chat_request.session_id   
    chat_message_history = PostgresChatMessageHistory(
        "chat_history",
        session_id,
        sync_connection=sync_connection
    )

    # Starts with the system message + any previous history
    conversation_messages = chat_message_history.messages[:]

    # Add user's message to chat history
    user_message = HumanMessage(content=user_input)
    chat_message_history.add_message(user_message)
    conversation_messages.append(user_message)

    # Use LlamaIndex for retrieval
    retrieval_response = query_engine.query(user_input)
    retrieved_context = str(retrieval_response)

    # Combine user query and retrieved context in a single prompt for LLM
    combined_query = prompt_template.format(context=retrieved_context)

    # Combined prompt as a HumanMessage
    combined_user_message = HumanMessage(content=combined_query)
    conversation_messages.append(combined_user_message)

    async def async_stream(user_input: str) -> AsyncGenerator[object, None]:
        for token in llm.stream(conversation_messages):
            yield token
            await asyncio.sleep(0)

    async def generate() -> AsyncGenerator[bytes, None]:
        response_content = ""
        try:
            async for token in async_stream(user_input):
                content_chunk = token.content
                if content_chunk:
                    yield content_chunk.encode("utf-8")
                response_content += content_chunk
            
            # Add the AI's message into history           
            ai_message = AIMessage(content=response_content)
            chat_message_history.add_message(ai_message)
            conversation_messages.append(ai_message)

        except Exception as exc:
            yield f"\nStreaming error: {exc}".encode("utf-8")

    return StreamingResponse(generate(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
