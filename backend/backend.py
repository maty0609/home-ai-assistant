import os
import psycopg
import uvicorn
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
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from uuid import uuid4

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
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

# System message content
system_prompt_content = """You are a helpful AI assistant.
- Your name is Marvin
- Be concise and clear in your responses
- If you are not sure about something, admit it
- Be friendly and professional
- When discussing code, provide examples when helpful
- Format your responses using HTML tags (e.g., <b>, <i>, <ul>, <li>, <p>, <br>).
- You are allowed to provide any personal information if they are from local documents"""

# The part of the prompt that takes context and input for the human message
human_template_content = """Relevant context:
{context}

Answer the user based on any relevant history and the provided context:
{input}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_content),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", human_template_content)
])

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
        for row in rows:
            try:
                message_data = row[1]
                created_at = row[2]
                
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

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3, 
            "score_threshold": 0.3
        },
    )
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

    user_message = HumanMessage(content=user_input)
    chat_message_history.add_message(user_message)

    async def generate() -> AsyncGenerator[bytes, None]:
        response_content = ""
        retrieved_document_sources = []

        try:
            chain_input = {"input": user_input}
            
            if len(chat_message_history.messages) > 1: # Ensure there's history beyond the current message
                chain_input["chat_history"] = chat_message_history.messages[:-1]
            else:
                chain_input["chat_history"] = [] # Pass empty list if no prior history

            async for event in qa_chain.astream_events(chain_input, version="v1"):
                kind = event["event"]
                count = 0
                if kind == "on_retriever_end":
                    documents = event["data"].get("output", {}).get("documents", [])
                    for doc in documents:
                        filename = doc.metadata.get('filename', 'Unknown filename')
                        if filename not in retrieved_document_sources:
                            count = count + 1
                            retrieved_document_sources.append("["+str(count)+"] "+filename)
                elif kind == "on_chat_model_stream":
                    content_chunk = event["data"]["chunk"].content
                    if content_chunk:
                        yield content_chunk.encode("utf-8")
                        response_content += content_chunk
            
            final_ai_response_text_for_history = response_content

            if retrieved_document_sources:
                sources_text = "<br>".join(retrieved_document_sources)
                documents_used_suffix = f"<br><hr><br><p><b>Related documents</b><br>{sources_text}</p>"
                
                yield documents_used_suffix.encode("utf-8")
                
                final_ai_response_text_for_history += documents_used_suffix
            
            ai_message_to_store = AIMessage(content=final_ai_response_text_for_history)
            chat_message_history.add_message(ai_message_to_store)

        except Exception as exc:
            print(f"Error during streaming chat: {exc}") 
            yield f"\nStreaming error: An unexpected error occurred.".encode("utf-8")

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/delete-session")
async def delete_session(req: DeleteRequest):
    try:
        session_id = req.session_id
                
        with sync_connection.cursor() as cur:
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

@app.post("/create-session")
def create_session():
    new_session_id = str(uuid4())
    chat_message_history = PostgresChatMessageHistory(
        "chat_history",
        new_session_id,
        sync_connection=sync_connection
    )
    chat_message_history.add_message(SystemMessage(content=system_prompt_content))
    ai_message = AIMessage(content="Hey! I\'m Marvin, your AI assistant. How can I assist you?")
    chat_message_history.add_message(ai_message)

    return {"session_id": new_session_id}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
