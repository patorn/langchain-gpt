import os
import sqlite3
import requests
import tempfile
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDataBaseTool
import chromadb
# Replace in-memory history import with persistent SQLAlchemy history
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine, text

load_dotenv() 

# Setup Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

# Create sample database for SQL tool
def create_sample_database():
    """Create a sample SQLite database with some demo data"""
    db_path = "sample_data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)
    
    # Insert sample data
    sample_employees = [
        (1, "Alice Johnson", "Engineering", 85000, "2022-01-15"),
        (2, "Bob Smith", "Marketing", 65000, "2021-03-10"),
        (3, "Carol Davis", "Engineering", 92000, "2020-11-20"),
        (4, "David Wilson", "Sales", 58000, "2023-02-01"),
        (5, "Eve Brown", "HR", 72000, "2021-08-30")
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO employees VALUES (?, ?, ?, ?, ?)", sample_employees)
    
    # Create products table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)
    
    sample_products = [
        (1, "Laptop Pro", "Electronics", 1299.99, 25),
        (2, "Wireless Mouse", "Electronics", 29.99, 150),
        (3, "Office Chair", "Furniture", 249.99, 45),
        (4, "Standing Desk", "Furniture", 399.99, 20),
        (5, "Coffee Maker", "Appliances", 89.99, 60)
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?)", sample_products)
    
    conn.commit()
    conn.close()
    return db_path

# Load and split documents (example: local text file)
def load_docs(path="example.txt"):
    # Create example file if it doesn't exist
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("""
            LangChain Documentation
            
            LangChain is a framework for developing applications powered by language models. 
            It enables applications that are context-aware and can reason about their environment.
            
            Key Components:
            1. LLMs and Chat Models - The core language model interface
            2. Prompts - Templates and strategies for formatting inputs
            3. Chains - Sequences of calls to LLMs or other utilities
            4. Agents - Use LLMs to decide which actions to take
            5. Memory - Persist state between calls
            6. Document Loaders - Load data from various sources
            
            RAG (Retrieval Augmented Generation):
            RAG combines retrieval and generation to provide more accurate and contextual responses.
            It works by first retrieving relevant documents, then using those documents to generate responses.
            
            Vector Stores:
            Vector stores enable semantic search by storing document embeddings.
            Common vector stores include Chroma, FAISS, and Pinecone.
            """)
    
    loader = TextLoader(path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)

# Initialize vector store with embeddings
def create_vectorstore(documents, collection_name: str = "langchain_docs", persist_directory: str = "chroma_store"):
    """Create (or load) a persistent Chroma vectorstore using chromadb PersistentClient.
    The new langchain_chroma wrapper does not expose .persist(); persistence is handled
    automatically when using PersistentClient. Returns (vectorstore, newly_ingested_flag).
    """
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create / connect persistent chromadb client
    client = chromadb.PersistentClient(path=str(persist_path))

    # Build vectorstore wrapper (collection auto-created if missing)
    vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    newly_ingested = False
    try:
        existing_count = vs._collection.count()
    except Exception:
        existing_count = 0

    if existing_count == 0 and documents:
        print(f"🆕 Ingesting {len(documents)} documents into collection '{collection_name}' ...")
        vs.add_documents(documents)
        newly_ingested = True
    else:
        print(f"ℹ️  Using existing collection '{collection_name}' with {existing_count} embeddings (no re-ingest).")

    # Verification
    try:
        sample = vs.get(limit=3).get("ids", [])
        print(f"🔍 Vectorstore verification sample ids: {sample}")
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")

    return vs, newly_ingested

# Setup a simple QA chain on top of vector store retrieval
def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

# Tool 1: Calculator
def calculator_fn(query: str) -> str:
    """Safe calculator tool that evaluates mathematical expressions"""
    try:
        # Import math module for mathematical functions
        import math
        
        # Only allow safe mathematical operations
        allowed_names = {
            # Basic functions
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            # Mathematical constants
            'pi': math.pi, 'e': math.e,
            # Mathematical functions
            'sqrt': math.sqrt,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'log': math.log, 'log10': math.log10,
            'exp': math.exp,
            'floor': math.floor, 'ceil': math.ceil,
            'degrees': math.degrees, 'radians': math.radians,
        }
        
        result = str(eval(query, {"__builtins__": {}}, allowed_names))
        return f"Calculation result: {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

calculator_tool = Tool(
    name="Calculator",
    func=calculator_fn,
    description="Useful for doing mathematical calculations. Input should be a valid mathematical expression like 'sqrt(144)' or '25 * 4 + sin(pi/2)'."
)

# Tool 2: Web Search (enhanced mock implementation)
def web_search_fn(query: str) -> str:
    """Enhanced web search simulation with some real-world context"""
    # In a real implementation, you would use services like:
    # - Google Custom Search API
    # - DuckDuckGo API
    # - Serper API
    
    mock_responses = {
        "weather": "Current weather: Sunny, 22°C in San Francisco. Light breeze from the west.",
        "news": "Latest tech news: AI developments continue to accelerate with new LLM releases.",
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "langchain": "LangChain is a framework for developing applications powered by language models.",
        "gemini": "Google's Gemini is a family of multimodal large language models.",
        "stock": "Stock market update: Tech stocks are showing mixed performance today."
    }
    
    query_lower = query.lower()
    for keyword, response in mock_responses.items():
        if keyword in query_lower:
            return f"Web search results for '{query}': {response}"
    
    return f"Web search results for '{query}': No specific information found, but here are general search results related to your query."

web_search_tool = Tool(
    name="WebSearch",
    func=web_search_fn,
    description="Useful for searching the web for current information, news, weather, or general knowledge."
)

# Tool 3: SQL Database Tool
def create_sql_tool(db_path: str):
    """Create SQL database tool"""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        
        def sql_query_fn(query: str) -> str:
            """Execute SQL query on the database"""
            try:
                # Basic safety check - only allow SELECT statements
                if not query.strip().upper().startswith('SELECT'):
                    return "Error: Only SELECT queries are allowed for safety reasons."
                
                result = db.run(query)
                return f"SQL Query Result:\n{result}"
            except Exception as e:
                return f"SQL Error: {e}"
        
        return Tool(
            name="SQLDatabase",
            func=sql_query_fn,
            description=f"""Useful for querying the SQL database. 
            Available tables: employees (id, name, department, salary, hire_date), 
            products (id, name, category, price, stock).
            Only SELECT queries are allowed. Input should be a valid SQL SELECT statement."""
        )
    except Exception as e:
        def fallback_sql(query: str) -> str:
            return f"SQL tool unavailable: {e}"
        
        return Tool(
            name="SQLDatabase",
            func=fallback_sql,
            description="SQL database tool (currently unavailable)"
        )

# Main agent initialization
def main():
    print("🚀 Initializing LangChain RAG + Agent System with Gemini...")
    
    # Create sample database
    print("📊 Setting up sample database...")
    db_path = create_sample_database()
    
    # Load docs and create vectorstore for RAG
    print("📚 Loading documents and creating vector store...")
    docs = load_docs()
    vectorstore, ingested = create_vectorstore(docs)
    if ingested:
        print("✅ Documents embedded & stored in Chroma.")
    else:
        print("✅ Using previously stored embeddings (no new ingestion).")
    qa_chain = create_qa_chain(vectorstore)

    # Create all tools
    print("🔧 Setting up tools...")
    sql_tool = create_sql_tool(db_path)
    tools = [calculator_tool, web_search_tool, sql_tool]

    # Add document QA tool for RAG
    def doc_qa_func(query: str) -> str:
        try:
            result = qa_chain.invoke({"query": query})
            return f"Document Search Result: {result['result'] if 'result' in result else str(result)}"
        except Exception as e:
            return f"Document search error: {e}"
    
    doc_qa_tool = Tool(
        name="DocumentQA",
        func=doc_qa_func,
        description="Use this to search and answer questions based on the LangChain documentation. Good for questions about LangChain concepts, RAG, vector stores, etc."
    )
    tools.append(doc_qa_tool)

    # ---- Conversation memory (new API) ----
    # Persist chat history in a local SQLite database using SQLChatMessageHistory
    CHAT_HISTORY_DB_PATH = "chat_history.db"
    CHAT_HISTORY_CONN_STR = f"sqlite:///{CHAT_HISTORY_DB_PATH}"  # SQLAlchemy connection string

    # Ensure file exists (SQLAlchemy will create tables automatically on first use)
    Path(CHAT_HISTORY_DB_PATH).touch(exist_ok=True)

    def ensure_history_index(conn_str: str):
        """Ensure a simple composite index on message_store(session_id, created_at).
        Safe to call multiple times; skips if table or columns not yet present.
        """
        try:
            engine = create_engine(conn_str)
            with engine.begin() as conn:
                # Check table exists
                exists = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='message_store'" )).fetchone()
                if not exists:
                    print("ℹ️ Chat history table 'message_store' not created yet; will try again later.")
                    return
                # Inspect columns
                cols = {r[1] for r in conn.execute(text("PRAGMA table_info(message_store)")).fetchall()}
                required = {"session_id", "created_at"}
                if not required.issubset(cols):
                    print("ℹ️ Required columns session_id/created_at not present; skipping index creation.")
                    return
                idx_name = "idx_message_store_session_created_at"
                idx_exists = conn.execute(text("SELECT name FROM sqlite_master WHERE type='index' AND name=:i"), {"i": idx_name}).fetchone()
                if not idx_exists:
                    conn.execute(text(f"CREATE INDEX {idx_name} ON message_store(session_id, created_at)"))
                    print(f"📈 Created index {idx_name} on message_store(session_id, created_at)")
                else:
                    print(f"ℹ️ Index {idx_name} already exists; skipping creation.")
        except Exception as e:
            print(f"⚠️ Could not ensure history index: {e}")

    # Ensure index before starting session usage
    ensure_history_index(CHAT_HISTORY_CONN_STR)

    def get_session_history(session_id: str) -> SQLChatMessageHistory:
        """Return a persistent chat history object for the given session id."""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=CHAT_HISTORY_CONN_STR,
            table_name="message_store",
        )
    # ---------------------------------------

    # Debug / utility: format and expose conversation history as a tool
    def format_history(session_id: str = "default") -> str:
        history = get_session_history(session_id)
        msgs = history.messages
        if not msgs:
            return "(History is empty)"
        lines = []
        for i, m in enumerate(msgs, 1):
            role = getattr(m, 'type', 'message')
            content = m.content if isinstance(m.content, str) else str(m.content)
            lines.append(f"{i}. {role}: {content}")
        return "\n".join(lines)

    def show_memory_tool_fn(_: str) -> str:
        # Prefix to make it clear it's final output (helps parsing)
        return "Conversation History:\n" + format_history()

    memory_tool = Tool(
        name="ShowMemory",
        func=show_memory_tool_fn,
        description="Displays the full conversation history so far (persisted in SQLite). Use when the user asks to see conversation history, prior messages, or memory."
    )
    tools.append(memory_tool)

    # Create enhanced react prompt template
    template = """You are a helpful AI assistant with access to multiple tools. Answer questions as accurately as possible using the available tools. If the user asks to see the conversation history, call the ShowMemory tool.

Available tools:
{tools}

Use this format:

Question: the input question you must answer
Thought: think about what you need to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Examples of what you can do:
- Mathematical calculations: "What is 25 * 4 + sqrt(16)?"
- Database queries: "Show me all employees in the Engineering department"
- Document search: "What is RAG in LangChain?"
- Web search: "What's the latest news about Python?"
- Show history: "Show me the conversation so far" / "history"

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

    # Create the agent
    print("🤖 Creating agent...")
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor (no legacy memory argument)
    base_agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )

    # Wrap with message history for stateful conversations (now persistent)
    agent_executor = RunnableWithMessageHistory(
        base_agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output"
    )

    print("✅ Agent is ready! You can now ask questions.")
    print("\nExample queries you can try:")
    print("- 'Calculate the square root of 144'")
    print("- 'Show me all employees with salary > 70000'")
    print("- 'What is RAG in LangChain?'")
    print("- 'Search for recent AI news'")
    print("- 'What's the total value of all products in stock?'")
    print("\nType 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit", "bye"}:
            print("👋 Goodbye!")
            break
        if not query:
            continue
        # Direct shortcut to bypass agent parsing for history requests
        if query.lower() in {"history", "show history", "show memory", "/history", "/memory"}:
            print("\n📜 Conversation History (direct):\n")
            print(format_history("default"))
            print()
            continue
        try:
            print("\n🤔 Processing your request...\n")
            response = agent_executor.invoke(
                {"input": query},
                config={"configurable": {"session_id": "default"}}
            )
            print(f"\n🤖 Agent: {response['output']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            
    # Cleanup
    try:
        os.remove(db_path)
        print("🧹 Cleaned up temporary database.")
    except:
        pass

if __name__ == "__main__":
    main()
