import os
import pymysql
import json
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from uuid import uuid4
from jose import JWTError, jwt
from argon2 import PasswordHasher

load_dotenv()

password_hasher = PasswordHasher()
security = HTTPBearer()

class MySQLChatMessageHistory:    
    def __init__(self, table_name: str, session_id: str, sync_connection):
        self.table_name = table_name
        self.session_id = session_id
        self.connection = sync_connection
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        with self.connection.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    message JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session_id (session_id),
                    INDEX idx_created_at (created_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            self.connection.commit()
    
    @property
    def messages(self) -> List[BaseMessage]:
        with self.connection.cursor() as cur:
            cur.execute(
                f"SELECT message FROM {self.table_name} WHERE session_id = %s ORDER BY created_at ASC",
                (self.session_id,)
            )
            rows = cur.fetchall()
            
            messages = []
            for row in rows:
                try:
                    message_data = row[0]
                    if isinstance(message_data, str):
                        message_data = json.loads(message_data)
                    elif not isinstance(message_data, dict):
                        continue
                    
                    if 'type' in message_data and 'data' in message_data:
                        msg_type = message_data['type']
                        content = message_data['data'].get('content', '')
                    elif 'content' in message_data:
                        content = message_data['content']
                        msg_type = message_data.get('type', 'human')
                    else:
                        continue
                    
                    if msg_type == 'human':
                        messages.append(HumanMessage(content=content))
                    elif msg_type == 'ai':
                        messages.append(AIMessage(content=content))
                    elif msg_type == 'system':
                        messages.append(SystemMessage(content=content))
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                    print(f"Error parsing message: {e}")
                    continue
            
            return messages
    
    def add_message(self, message: BaseMessage):
        if isinstance(message, HumanMessage):
            message_data = {
                "type": "human",
                "data": {"content": message.content}
            }
        elif isinstance(message, AIMessage):
            message_data = {
                "type": "ai",
                "data": {"content": message.content}
            }
        elif isinstance(message, SystemMessage):
            message_data = {
                "type": "system",
                "data": {"content": message.content}
            }
        else:
            message_data = {
                "type": str(type(message).__name__).lower().replace("message", ""),
                "data": {"content": message.content}
            }
        
        with self.connection.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table_name} (session_id, message) VALUES (%s, %s)",
                (self.session_id, json.dumps(message_data))
            )
            self.connection.commit()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_URL'),
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    openai_api_version=os.getenv('AZURE_OPENAI_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY'),
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("EMB_OPENAI_URL"),
    azure_deployment="ttext-embedding-3-large",
    api_version="2024-08-01-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
)

db_url = os.getenv('DB_MARIADB_URL')
if db_url and (db_url.startswith('mysql://') or db_url.startswith('mariadb://')):
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    db_config = {
        'host': parsed.hostname or 'localhost',
        'port': parsed.port or 3306,
        'user': parsed.username,
        'password': parsed.password,
        'database': parsed.path.lstrip('/'),
        'charset': 'utf8mb4',
        'autocommit': False
    }
    sync_connection = pymysql.connect(**db_config)
else:
    sync_connection = pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', 3306)),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        charset='utf8mb4',
        autocommit=False
    )

table_name = "chat_history"
vector_store = Chroma(
    persist_directory=os.getenv("CHROMADB_PATH"),
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

system_prompt_content = """You are a helpful AI assistant.
- Your name is Marvin
- Be concise and clear in your responses
- If you are not sure about something, admit it
- Be friendly and professional
- When discussing code, provide examples when helpful
- Format your responses using HTML tags (e.g., <b>, <i>, <ul>, <li>, <p>, <br>).
- You are allowed to provide any personal information if they are from local documents"""

human_template_content = """Relevant context:
{context}

Answer the user based on any relevant history and the provided context:
{input}"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt_content),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", human_template_content)
])

class UserCreate(BaseModel):
    email: str
    name: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: int
    email: str
    name: str

class Token(BaseModel):
    access_token: str
    token_type: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    id: int
    email: str
    name: str

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


def verify_password(plain_password, hashed_password):
    try:
        password_hasher.verify(hashed_password, plain_password)
        return True
    except:
        return False

def get_password_hash(password):
    return password_hasher.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("SECRET_KEY"), algorithm="HS256")
    return encoded_jwt

def get_user_by_email(email: str):
    with sync_connection.cursor() as cur:
        cur.execute("SELECT id, email, name, hashed_password FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        if user:
            return {
                "id": user[0],
                "email": user[1],
                "name": user[2],
                "hashed_password": user[3]
            }
    return None

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, os.getenv("SECRET_KEY"), algorithms=["HS256"])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/auth/register", response_model=User)
async def register_user(user: UserCreate):
    existing_user = get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    hashed_password = get_password_hash(user.password)
    
    with sync_connection.cursor() as cur:
        cur.execute(
            "INSERT INTO users (email, name, hashed_password) VALUES (%s, %s, %s)",
            (user.email, user.name, hashed_password)
        )
        user_id = cur.lastrowid
        sync_connection.commit()
        
        cur.execute("SELECT id, email, name FROM users WHERE id = %s", (user_id,))
        new_user = cur.fetchone()
    
    return {
        "id": new_user[0],
        "email": new_user[1],
        "name": new_user[2]
    }

@app.post("/auth/login", response_model=LoginResponse)
async def login_user(user_credentials: UserLogin):
    user = authenticate_user(user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "id": user["id"],
        "email": user["email"],
        "name": user["name"]
    }

@app.get("/auth/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["id"],
        "email": current_user["email"],
        "name": current_user["name"]
    }

@app.get("/sessions")
def list_sessions(current_user: dict = Depends(get_current_user)):
    with sync_connection.cursor() as cur:
        cur.execute("""
            SELECT session_id, message, created_at
            FROM (
                SELECT 
                    session_id,
                    message,
                    created_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY session_id 
                        ORDER BY 
                            CASE WHEN message LIKE '%"type": "human"%' THEN 0 ELSE 1 END ASC,
                            created_at ASC
                    ) as rn
                FROM chat_history
            ) AS SessionTitles
            WHERE rn = 1
            ORDER BY created_at DESC
        """)
        rows = cur.fetchall()
        
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
                
                # Parse JSON message
                content = ""
                if isinstance(message_data, str):
                    try:
                        message_json = json.loads(message_data)
                        content = message_json.get('data', {}).get('content', '')
                    except json.JSONDecodeError:
                        content = message_data
                elif isinstance(message_data, dict):
                    content = message_data.get('data', {}).get('content', '')
                
                # Fallback if content is empty
                if not content:
                    content = "New Conversation"

                session_data = {
                    "session_id": row[0],
                    "last_message": content,
                    "message_type": "human",
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
                    
            except (AttributeError, TypeError, Exception) as e:
                print(f"Error processing session: {e}")
                continue
                
    return {"sessions": sessions_by_period}

@app.post("/history", response_model=HistoryResponse)
async def history_endpoint(req: HistoryRequest, current_user: dict = Depends(get_current_user)):
    try:
        session_id = req.session_id
        
        chat_message_history = MySQLChatMessageHistory(
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
async def stream_chat(chat_request: ChatRequest, current_user: dict = Depends(get_current_user)) -> StreamingResponse:
    user_input = chat_request.user_input
    session_id = chat_request.session_id
    chat_message_history = MySQLChatMessageHistory(
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
            
            if len(chat_message_history.messages) > 1:
                chain_input["chat_history"] = chat_message_history.messages[:-1]
            else:
                chain_input["chat_history"] = []

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
async def delete_session(req: DeleteRequest, current_user: dict = Depends(get_current_user)):
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
    chat_message_history = MySQLChatMessageHistory(
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
