# LangChain RAG + Agent Prototype with Gemini

## Overview
This is a comprehensive single-file prototype demonstrating:
- **RAG (Retrieval Augmented Generation)** with document search
- **Multi-tool Agent** with calculator, web search, and SQL database
- **Local execution** with Google's Gemini model
- **Conversational memory** for context retention

## Features

### 🤖 Agent Tools
1. **Calculator** - Safe mathematical calculations with advanced functions
2. **Web Search** - Simulated web search (can be replaced with real APIs)
3. **SQL Database** - Query a sample SQLite database with employees and products
4. **Document QA** - RAG-based question answering on LangChain documentation

### 📊 Sample Database
The prototype creates a sample SQLite database with:
- **employees** table (id, name, department, salary, hire_date)
- **products** table (id, name, category, price, stock)

### 🔍 RAG System
- Loads and indexes LangChain documentation
- Uses Google's embedding model for semantic search
- Retrieves relevant context for questions

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get Google API Key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy `.env.example` to `.env` and add your key

3. **Run the prototype:**
   ```bash
   python singlefile_prototype.py
   ```

## Example Queries

### Mathematical Calculations
- "Calculate the square root of 144"
- "What is 25 * 4 + sin(pi/2)?"
- "Find the sum of 1 + 2 + 3 + ... + 100"

### Database Queries
- "Show me all employees in the Engineering department"
- "What's the average salary by department?"
- "List all products with price > 100"
- "What's the total value of all products in stock?"

### Document Search (RAG)
- "What is RAG in LangChain?"
- "Explain vector stores"
- "How do LangChain agents work?"
- "What are the key components of LangChain?"

### Web Search
- "Search for recent AI news"
- "What's the weather like?"
- "Find information about Python programming"

## Architecture

```
User Query → Agent → Tool Selection → Tool Execution → Response
                ↓
            Memory Update
                ↓
         Context Retention
```

### Agent Flow
1. **Question Analysis** - Agent analyzes the user query
2. **Tool Selection** - Chooses appropriate tool(s)
3. **Tool Execution** - Executes selected tool with parameters
4. **Result Processing** - Processes and formats results
5. **Response Generation** - Provides final answer to user

## Customization

### Adding Real Web Search
Replace the mock web search with real APIs:
```python
# Example with DuckDuckGo
from duckduckgo_search import ddg

def web_search_fn(query: str) -> str:
    results = ddg(query, max_results=3)
    return str(results)
```

### Adding More Database Tables
Extend the `create_sample_database()` function:
```python
cursor.execute("""
    CREATE TABLE your_table (
        id INTEGER PRIMARY KEY,
        data TEXT NOT NULL
    )
""")
```

### Custom Document Sources
Modify `load_docs()` to load from different sources:
```python
# Load from URL, PDF, etc.
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
```

## Security Notes
- SQL queries are restricted to SELECT statements only
- Calculator uses safe evaluation with limited functions
- In production, implement proper input validation and sanitization

## Troubleshooting

### Common Issues
1. **Missing API Key**: Ensure `GOOGLE_API_KEY` is set in `.env`
2. **Import Errors**: Install all requirements with `pip install -r requirements.txt`
3. **Database Errors**: Check file permissions for SQLite database creation

### Debug Mode
Set `verbose=True` in AgentExecutor to see detailed execution steps.

## License
MIT License - Feel free to modify and extend for your needs!
# langchain-gpt
