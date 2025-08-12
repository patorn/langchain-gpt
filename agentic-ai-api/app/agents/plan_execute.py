from typing import List, Tuple

# Third-party
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser

# Local
from ..core.vectorstore import get_vectorstore
from ..core.config import settings

__all__ = [
    "build_tools",
    "create_agent",
    "generate_plan",
    "plan_then_execute",
]

# -----------------------------
# Configuration constants
# -----------------------------
DEFAULT_MIN_STEPS: int = 8
AGENT_MAX_ITERATIONS: int = 10
PLANNER_MAX_TOKENS: int = 1024
AGENT_MAX_TOKENS: int = 2048

# -----------------------------
# Prompts
# -----------------------------
_plan_template = (
    "You are a planning agent. Carefully break the user goal into at least {min_steps} "
    "small, verifiable steps using available tools.\n"
    "- Make steps specific and atomic (retrieve, compute, synthesize separated).\n"
    "- Keep each step concise (one action per step).\n"
    "Return ONLY a numbered list of steps.\n"
    "User Goal: {goal}\n"
)

_react_template = (
    "You are an execution agent with tools.\n"
    "Tools:\n{tools}\nTool names: {tool_names}\n"
    "Follow EXACT format when using tools:\n"
    "Thought: <reasoning>\n"
    "Action: <one tool name>\n"
    "Action Input: <input string>\n"
    "After a tool call you will be given its result as Observation and you continue. "
    "When you know the answer output ONLY:\n"
    "Final Answer: <answer text>\n"
    "Begin!\nQuestion: {input}\n{agent_scratchpad}"
)

# -----------------------------
# Internal caches
# -----------------------------
_planner_chain = None  # lazy-initialized planning chain


def _get_planner_chain():
    """Build or return a cached planning chain.

    Returns a composed chain: PromptTemplate -> LLM -> StrOutputParser.
    """
    global _planner_chain
    if _planner_chain is None:
        plan_prompt = PromptTemplate(template=_plan_template, input_variables=["goal", "min_steps"])
        llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0, max_output_tokens=PLANNER_MAX_TOKENS)
        _planner_chain = plan_prompt | llm | StrOutputParser()
    return _planner_chain


# -----------------------------
# Tools and Agent construction
# -----------------------------

def build_tools(vs=None) -> List[Tool]:
    """Create the list of Tool objects used by the agent.

    Includes:
    - Calculator: safe eval with sqrt
    - DocSearch: RetrievalQA over the vector store
    """
    from math import sqrt

    def calc(expr: str) -> str:
        try:
            return str(eval(expr, {"__builtins__": {}}, {"sqrt": sqrt}))
        except Exception as e:
            return f"Calc error: {e}"

    vs = vs or get_vectorstore()
    qa = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0),
        chain_type="stuff",
        retriever=vs.as_retriever(),
    )

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
    """Create the ReAct agent with tools and prompt.

    Returns a configured AgentExecutor with slightly higher iteration limit
    and parsing error handling enabled.
    """
    tools = build_tools()
    llm = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0, max_output_tokens=AGENT_MAX_TOKENS)
    prompt = PromptTemplate(template=_react_template, input_variables=["input", "agent_scratchpad", "tools", "tool_names"])
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=AGENT_MAX_ITERATIONS, handle_parsing_errors=True)


# -----------------------------
# Planning + Execution Orchestration
# -----------------------------

def _parse_plan_lines(raw: str) -> List[str]:
    """Parse numbered plan text into a list of step strings."""
    steps: List[str] = []
    for line in raw.splitlines():
        t = line.strip()
        if not t:
            continue
        if t[0].isdigit():
            # remove leading numbering like '1.' or '2)'
            if '.' in t:
                t = t.split('.', 1)[1].strip()
            elif ')' in t:
                t = t.split(')', 1)[1].strip()
        steps.append(t)
    return steps


def generate_plan(goal: str, min_steps: int = DEFAULT_MIN_STEPS) -> List[str]:
    """Generate a list of atomic steps for a goal using the planner chain."""
    raw = _get_planner_chain().invoke({"goal": goal, "min_steps": min_steps})
    steps = _parse_plan_lines(raw)
    # Ensure minimum steps by padding with a generic synthesis if needed
    while len(steps) < min_steps:
        steps.append("Synthesize findings so far and identify any missing evidence.")
    return steps


def _normalize_final_answer(text: str) -> str:
    """Strip leading 'Final Answer:' prefix if present."""
    if text.lower().startswith("final answer:"):
        return text.split(':', 1)[1].strip()
    return text


def plan_then_execute(goal: str, agent: AgentExecutor) -> Tuple[str, List[str]]:
    """Plan the task into steps, execute each step via the ReAct agent, then synthesize.

    Returns (final_answer, steps).
    """
    steps = generate_plan(goal, min_steps=DEFAULT_MIN_STEPS)
    if not steps:
        steps = ["Analyze the question and answer directly using available tools if needed."]

    notes: List[str] = []
    for idx, step in enumerate(steps, 1):
        sub_input = (
            f"Step {idx}: {step}\n"
            f"Context so far: {' '.join(notes[-3:])}\n"
            f"Original Goal: {goal}\n"
            f"Take your time and use tools as needed."
        )
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
        synth_input = (
            "Produce a thorough final answer to the original goal using the step notes. "
            "Be concise but complete; verify important facts from the notes.\n"
            f"Goal: {goal}\nNotes: {notes}"
        )
        synth = agent.invoke({"input": synth_input})
        final_answer = synth.get("output", str(synth))
    else:
        final_answer = notes[-1]

    return _normalize_final_answer(final_answer), steps
