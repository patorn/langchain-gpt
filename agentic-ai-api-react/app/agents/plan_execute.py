from typing import List
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from ..core.vectorstore import get_vectorstore
from ..core.config import settings

_plan_template = """You are a planning agent. Break the user goal into ordered steps using available tools.
Return steps as a numbered list, concise.
User Goal: {goal}
"""

_react_template = """You are an execution agent with tools.
Tools:\n{tools}\nTool names: {tool_names}
Follow EXACT format when using tools:
Thought: <reasoning>
Action: <one tool name>
Action Input: <input string>
After a tool call you will be given its result as Observation and you continue. When you know the answer output ONLY:
Final Answer: <answer text>
Begin!\nQuestion: {input}\n{agent_scratchpad}"""


def build_tools(vs=None) -> List[Tool]:
    from math import sqrt
    def calc(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {"sqrt": sqrt}))
        except Exception as e:
            return f"Calc error: {e}"
    vs = vs or get_vectorstore()
    qa = RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0), chain_type="stuff", retriever=vs.as_retriever())
    def doc_search(q: str) -> str:
        try:
            r = qa.invoke({"query": q})
            return r.get("result", str(r))
        except Exception as e:
            return f"Doc search error: {e}"
    return [
        Tool(name="Calculator", func=calc, description="Evaluate simple sqrt expressions."),
        Tool(name="DocSearch", func=doc_search, description="Search embedded documents for answers."),
    ]


def create_agent() -> AgentExecutor:
    tools = build_tools()
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
    prompt = PromptTemplate(template=_react_template, input_variables=["input", "agent_scratchpad", "tools", "tool_names"])
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=4, handle_parsing_errors=True)
