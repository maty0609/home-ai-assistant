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
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

class DeleteRequest(BaseModel):
    session_id: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/sessions")
def list_sessions():
    with sync_connection.cursor() as cur:
        # Get the last message per session with creation date
        cur.execute("""
            WITH LastMessages AS (
                SELECT DISTINCT ON (session_id) 
                    session_id,
                    message,
                    created_at,
                    ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at DESC) as rn
                FROM chat_history
                ORDER BY session_id, created_at DESC
            )
            SELECT session_id, message, created_at
            FROM LastMessages
            WHERE rn = 1
            ORDER BY created_at DESC;
        """)
        rows = cur.fetchall()
        
        # Group sessions by time period
        now = datetime.now(ZoneInfo("UTC"))
        thirty_days_ago = now - timedelta(days=30)
        
        sessions_by_period = {
            "Last 30 days": []
        }
        
        # Extract content and group by time period
        for row in rows:
            try:
                message_data = row[1]
                created_at = row[2]
                
                # Ensure created_at is timezone-aware
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=ZoneInfo("UTC"))
                
                if isinstance(message_data, dict):
                    content = message_data.get('data', {}).get('content', '')
                else:
                    content = str(message_data) if message_data is not None else ""
                
                session_data = {
                    "session_id": row[0],
                    "last_message": content,
                    "message_type": "ai",
                    "created_at": created_at.isoformat()
                }
                
                # Group by time period
                if created_at >= thirty_days_ago:
                    sessions_by_period["Last 30 days"].append(session_data)
                else:
                    # Group older sessions by month
                    month_key = created_at.strftime("%B %Y")
                    if month_key not in sessions_by_period:
                        sessions_by_period[month_key] = []
                    sessions_by_period[month_key].append(session_data)
                    
            except (AttributeError, TypeError) as e:
                print(f"Error processing session: {e}")
                continue
                
    return {"sessions": sessions_by_period}

@app.post("/history", response_model=HistoryResponse)
async def history_endpoint(req: HistoryRequest):
    try:
        session_id = req.session_id
        
        chat_message_history = PostgresChatMessageHistory(
            "chat_history",
            session_id,
            sync_connection=sync_connection
        )
 
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

@app.post("/delete-session")
async def delete_session(req: DeleteRequest):
    try:
        session_id = req.session_id
                
        with sync_connection.cursor() as cur:
            # First check if the session exists
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE session_id = %s", (session_id,))
            count = cur.fetchone()[0]
            
            if count == 0:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Delete the session
            cur.execute("DELETE FROM chat_history WHERE session_id = %s", (session_id,))
            deleted_count = cur.rowcount
            sync_connection.commit()
                        
        return {"status": "success", "deleted_count": deleted_count}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
