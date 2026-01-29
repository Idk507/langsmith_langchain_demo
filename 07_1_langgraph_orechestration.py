# User Input
      
#       ↓

# Planner Agent
  
#       ↓

# ReAct / AoT Executor (with web search middleware)
   
#       ↓

# Final Answer

import os
import uuid
import json
from typing import TypedDict, Annotated, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("LangSmith project:", PROJECT_NAME)

# LLM Wrapper 
def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=1024) -> str:
    with trace(
        name="llm_call",
        project_name=PROJECT_NAME,
        metadata={"model": model},
        tags=["llm"],
    ):
        resp = hf_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        result = resp.choices[0].message.content
        print(f"[LLM Response]: {result[:200]}...")  # Debug
        return result
# Web Search Middleware (DuckDuckGo)
def web_search(query: str, k: int = 3) -> str:
    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["tool", "web"],
    ):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    title = r.get("title","")
                    snippet = r.get("body","")
                    link = r.get("href","")
                    results.append(f"{title}\n{snippet}\nSource: {link}")
            output = "\n\n".join(results) if results else "No results found."
            print(f"[Web Search]: Found {len(results)} results")  # Debug
            return output
        except Exception as e:
            print(f"[Web Search Error]: {e}")
            return f"Search failed: {str(e)}"


def extract_json(text: str) -> Any:
    """Extract JSON from text that might contain markdown or extra text."""
    text = text.strip()
    
    # Remove markdown code blocks
    if "```json" in text.lower():
        parts = text.split("```")
        for part in parts:
            if part.strip().startswith(("json", "[")):
                text = part.replace("json", "").strip()
                break
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON array or object
    json_pattern = r'(\[.*?\]|\{.*?\})'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Last resort: return as single-item list
    print(f"[JSON PARSE WARNING] Could not parse JSON, using fallback")
    return [text]


# Orchestration State
class OrchestrationState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: List[str] | None
    current_step: int
    observations: List[str]
    final_answer: str | None
    user_id: str

# Planner Node

def planner_node(state: OrchestrationState) -> Dict[str, Any]:
    """Decompose user query into actionable steps."""
    print("\n" + "="*70)
    print("PLANNER AGENT")
    print("="*70)
    
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content
    print(f"User Query: {user_query}")

    prompt = f"""You are a planning agent. Break down the user's question into 2-4 clear, actionable research steps.

User Question: {user_query}

Return ONLY a JSON array of strings (no markdown, no explanations).

Example format:
["Search for information about X", "Analyze the relationship between Y and Z", "Summarize findings"]

Your JSON array:"""

    raw = call_llm(prompt, max_new_tokens=512)
    
    try:
        plan = extract_json(raw)
        if not isinstance(plan, list) or len(plan) == 0:
            plan = [user_query]
    except Exception as e:
        print(f"[PLANNER ERROR]: {e}")
        plan = [user_query]
    
    print(f"\n Plan created with {len(plan)} steps:")
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")
    
    return {
        "plan": plan,
        "current_step": 0,
        "observations": []
    }


# Action Node (Tool Use)
def action_node(state: OrchestrationState) -> Dict[str, Any]:
    """Decide and execute an action for the current step."""
    print("\n" + "="*70)
    print(f"ACTION AGENT - Step {state['current_step'] + 1}/{len(state['plan'])}")
    print("="*70)
    
    step = state["plan"][state["current_step"]]
    print(f"Current Step: {step}")

    # Build context from previous observations
    context = ""
    if state["observations"]:
        context = f"\n\nPrevious findings:\n{state['observations'][-1][:300]}..."

    action_prompt = f"""You are an action agent. Decide what action to take for this step.

Current Step: {step}{context}

Available Actions:
1. WEB_SEARCH - Search the web for current information
2. ANSWER_FROM_KNOWLEDGE - Use your internal knowledge to answer

Return ONLY valid JSON (no markdown, no explanations):
{{"action": "WEB_SEARCH", "action_input": "your search query here"}}

OR

{{"action": "ANSWER_FROM_KNOWLEDGE", "action_input": "question to answer"}}

Your JSON:"""

    raw = call_llm(action_prompt, max_new_tokens=256)
    
    try:
        action_decision = extract_json(raw)
        if isinstance(action_decision, list):
            action_decision = action_decision[0] if action_decision else {}
        
        action_type = action_decision.get("action", "WEB_SEARCH")
        action_input = action_decision.get("action_input", step)
    except Exception as e:
        print(f"[ACTION PARSE ERROR]: {e}")
        action_type = "WEB_SEARCH"
        action_input = step
    
    print(f" Action: {action_type}")
    print(f" Input: {action_input}")

    # Execute the action
    if action_type == "WEB_SEARCH":
        observation = web_search(action_input, k=3)
    else:
        knowledge_prompt = f"""Answer this question concisely and accurately using your knowledge:

{action_input}

Provide a clear, factual answer:"""
        observation = call_llm(knowledge_prompt, max_new_tokens=512)
    
    print(f"Observation captured ({len(observation)} characters)")
    
    return {
        "observations": state["observations"] + [observation]
    }

# Thought Node (Reasoning)
def thought_node(state: OrchestrationState) -> Dict[str, Any]:
    """Reflect on the observation and what was learned."""
    print("\n" + "="*70)
    print("THOUGHT AGENT")
    print("="*70)
    
    step = state["plan"][state["current_step"]]
    observation = state["observations"][-1]

    thought_prompt = f"""You are a reasoning agent. Reflect on what was just learned.

Step Executed: {step}

Observation: {observation[:500]}...

Provide a brief reflection (2-3 sentences):
- What key information was discovered?
- Is this sufficient for the step?

Your reflection:"""

    thought = call_llm(thought_prompt, max_new_tokens=150).strip()
    print(f" Thought: {thought}")
    
    return {}


# Step Update Node
def update_step_node(state: OrchestrationState) -> Dict[str, Any]:
    """Move to the next step."""
    new_step = state["current_step"] + 1
    print(f"\n Moving to step {new_step + 1}/{len(state['plan'])}")
    return {"current_step": new_step}

# Router
def router(state: OrchestrationState) -> str:
    """Route to next action or final answer."""
    if state["current_step"] < len(state["plan"]):
        return "action"
    return "final_answer"


def final_answer_node(state: OrchestrationState) -> Dict[str, Any]:
    """Synthesize all observations into a final answer."""
    print("\n" + "="*70)
    print("FINAL ANSWER AGENT")
    print("="*70)
    
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content
    
    all_info = "\n\n---\n\n".join([
        f"Finding {i+1}:\n{obs}" 
        for i, obs in enumerate(state["observations"])
    ])
    
    print(f" Synthesizing {len(state['observations'])} observations...")

    final_prompt = f"""You are a synthesis agent. Create a comprehensive answer to the user's question using the research findings below.

Original Question: {user_query}

Research Findings:
{all_info}

Instructions:
- Provide a clear, well-structured answer
- Use the information from the findings
- Be accurate and comprehensive
- Do not mention the research process or agents
- Cite sources when relevant

Your final answer:"""

    answer = call_llm(final_prompt, max_new_tokens=1024)
    
    print(f" Final answer generated ({len(answer)} characters)")
    
    return {"final_answer": answer}


# Planner Agent  → decomposes

# Action Agent   → decides tool usage

# Tool           → executes

# Thought Agent  → reflects

# Final Agent    → synthesizes

# Build Orchestration Graph
def build_graph():
    """Build the orchestration graph."""
    
    checkpointer = MemorySaver()
    store = InMemoryStore()
    
    builder = StateGraph(OrchestrationState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("action", action_node)
    builder.add_node("thought", thought_node)
    builder.add_node("update_step", update_step_node)
    builder.add_node("final_answer", final_answer_node)
    
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "action")
    builder.add_edge("action", "thought")
    builder.add_edge("thought", "update_step")
    
    builder.add_conditional_edges(
        "update_step",
        router,
        {
            "action": "action",
            "final_answer": "final_answer",
        },
    )
    
    builder.add_edge("final_answer", END)
    
    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
    
    return graph
graph = build_graph()
# Run Orchestration
thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}

user_query = input("Ask a question: ")

initial_state: OrchestrationState = {
    "messages": [{"role": "user", "content": user_query}],
    "plan": None,
    "current_step": 0,
    "observations": [],
    "final_answer": None,
    "user_id": "user_1",
}
result = graph.invoke(initial_state, config)
result
print(result["final_answer"])

