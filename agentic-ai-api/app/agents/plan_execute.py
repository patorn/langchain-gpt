from typing import List
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
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

_planner_chain = None

def get_planner_chain():
    global _planner_chain
    if _planner_chain is None:
        plan_prompt = PromptTemplate(template=_plan_template, input_variables=["goal"])
        llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0)
        _planner_chain = plan_prompt | llm | StrOutputParser()
    return _planner_chain


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


def generate_plan(goal: str) -> List[str]:
    raw = get_planner_chain().invoke({"goal": goal})
    steps: List[str] = []
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        if t[0].isdigit():
            # remove leading numbering like '1.' or '2)'
            t = t.split('.', 1)[1].strip() if '.' in t else t.split(')', 1)[1].strip() if ')' in t else t
        steps.append(t)
    return steps


def plan_then_execute(goal: str, agent: AgentExecutor) -> tuple[str, List[str]]:
    steps = generate_plan(goal)
    notes: List[str] = []
    for idx, step in enumerate(steps, 1):
        sub_input = f"Step {idx}: {step}\nContext so far: {' '.join(notes[-3:])}\nOriginal Goal: {goal}"
        try:
            res = agent.invoke({"input": sub_input})
            out = res.get("output", str(res))
        except Exception as e:
            out = f"Execution error: {e}"
        notes.append(out)
        if out.lower().startswith("final answer:"):
            break
    # Synthesize final answer if last note not already a final answer
    if not notes[-1].lower().startswith("final answer:"):
        synth = agent.invoke({"input": f"Synthesize final answer for: {goal}. Use these step notes: {notes}"})
        final_answer = synth.get("output", str(synth))
    else:
        final_answer = notes[-1]
    # Normalize final answer prefix
    if final_answer.lower().startswith("final answer:"):
        final_answer = final_answer.split(':', 1)[1].strip()
    return final_answer, steps
