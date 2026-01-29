# 09_memory_long_term

# Purpose:
# - Add long-term memory to the agent
# - Store user preferences and learned facts
# - Retrieve memory before answering
# - Update memory after each run
import os
import uuid
import json
import re
from typing import TypedDict, Annotated, List, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

from huggingface_hub import InferenceClient
from ddgs import DDGS
assert os.getenv("HF_TOKEN"), "HF_TOKEN must be set"
assert os.getenv("LANGCHAIN_PROJECT"), "LANGCHAIN_PROJECT must be set"

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("LangSmith project:", PROJECT_NAME)
## LLM Wrapper
def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512) -> str:
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
        return resp.choices[0].message.content.strip()
## Web Search
# ---------------- Web Search ----------------

def web_search(query: str, k: int = 2) -> str:
    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["tool", "web"],
    ):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                results.append(
                    f"{r.get('title','')}\n{r.get('body','')}\nSource: {r.get('href','')}"
                )
        return "\n\n".join(results) if results else "No results found."

## JSON Extractor
def extract_json(text: str) -> Any:
    if not text:
        return []
    txt = text.strip()
    if txt.startswith("```"):
        parts = txt.split("```")
        for p in parts:
            c = p.strip()
            if c.startswith("[") or c.startswith("{"):
                txt = c
                break
    try:
        return json.loads(txt)
    except Exception:
        matches = re.findall(r'(\[.*?\]|\{.*?\})', txt, re.DOTALL)
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                pass
    return [txt]
## State Definition
class OrchestrationState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: Optional[List[str]]
    current_step: int
    observations: List[str]
    thought: Optional[str]
    final_answer: Optional[str]
    memory: Optional[str]
    memory_update_text: Optional[str]
    user_id: str
## Memory Helpers - FIXED
def get_long_term_memory(store: InMemoryStore, user_id: str) -> str:
    """Retrieve memory from store"""
    try:
        namespace = ("user_memory",)
        item = store.get(namespace, user_id)
        
        if item is None:
            return ""
        
        # Handle different possible return types
        if hasattr(item, 'value'):
            value = item.value
        else:
            value = item
        
        # If value is a dict with 'value' key, extract it
        if isinstance(value, dict) and 'value' in value:
            return value['value']
        
        # If value is already a string, return it
        if isinstance(value, str):
            return value
        
        # Fallback: convert to string
        return str(value) if value else ""
        
    except Exception as e:
        print(f"Error retrieving memory: {e}")
        return ""

def update_long_term_memory(store: InMemoryStore, user_id: str, memory_text: str):
    """Store memory in the correct format"""
    if memory_text:
        namespace = ("user_memory",)
        # Store as plain string, not wrapped in dict
        store.put(namespace, user_id, memory_text)
## Graph Nodes
def planner_node(state: OrchestrationState) -> Dict[str, Any]:
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content

    raw = call_llm(f"Break into at most 2 steps:\n{user_query}", max_new_tokens=128)
    plan = extract_json(raw)
    if not isinstance(plan, list):
        plan = [user_query]

    return {"plan": plan[:2], "current_step": 0, "observations": []}

def action_node(state: OrchestrationState) -> Dict[str, Any]:
    step = state["plan"][state["current_step"]]
    memory = state.get("memory", "") or ""

    # Build a memory augmentation prompt
    prompt = f"""
You are an Action Agent.

Step to execute:
{step}

Long-term Memory about the user:
{memory}

Try to answer the step using memory and your internal knowledge.
If you cannot answer confidently, return "INSUFFICIENT" only.
"""

    llm_answer = call_llm(prompt, max_new_tokens=1024)

    # If the model indicates insufficient knowledge, then do search
    if "insufficient" in llm_answer.lower():
        # Web search fallback
        search_prompt = f"Find information about: {step}"
        observation = web_search(search_prompt, k=2)
    else:
        # We treat the LLM answer as the observation
        observation = llm_answer

    return {"observations": state["observations"] + [observation]}
def thought_node(state: OrchestrationState) -> Dict[str, Any]:
    step = state["plan"][state["current_step"]]
    observation = state["observations"][-1]

    prompt = f"""
You are a Thought Agent.

Step executed:
{step}

Observation:
{observation[:300]}

Decide if this information is sufficient to complete the step.
Return ONLY one of: "Sufficient" or "Insufficient" with a short reason.

Your reasoning:
"""

    thought = call_llm(prompt, max_new_tokens=80).strip()
    return {"thought": thought}
def update_step_node(state: OrchestrationState) -> Dict[str, Any]:
    return {"current_step": state["current_step"] + 1}
def router(state: OrchestrationState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "action"
    return "final_answer"
def final_answer_node(state: OrchestrationState) -> Dict[str, Any]:
    last = state["messages"][-1]
    user_query = last["content"] if isinstance(last, dict) else last.content
    memory = state.get("memory") or ""

    all_info = "\n".join(state["observations"])
    prompt = f"""
You are a Synthesis Agent.

User Question:
{user_query}

Long-term Memory:
{memory}

New Collected Information:
{all_info}

Answer the question using memory and new info:
"""

    answer = call_llm(prompt, max_new_tokens=512)

    # Compose a memory-update prompt
    mem_prompt = f"""
You are maintaining long-term memory.

Existing memory:
{memory}

New information collected:
{all_info}

Extract:
- clear user preferences,
- user identity,
- stable facts about the user,
- anything worth remembering in future
- summarise and store the information above into the long-term memory.
Return ONLY the memory update text.
"""

    new_memory = call_llm(mem_prompt, max_new_tokens=256)
    return {"final_answer": answer, "memory_update_text": new_memory}
## Build Graph with Memory
def build_graph_with_memory():
    store = InMemoryStore()
    builder = StateGraph(OrchestrationState)

    # Define memory nodes as closures to capture the store variable
    def memory_load_node(state: OrchestrationState) -> Dict[str, Any]:
        """Load memory from store at the beginning"""
        mem = get_long_term_memory(store, state["user_id"])
        return {"memory": mem}

    def memory_persist_node(state: OrchestrationState) -> Dict[str, Any]:
        """Persist memory to store at the end"""
        memory_text = state.get("memory_update_text", "")
        if memory_text:
            update_long_term_memory(store, state["user_id"], memory_text)
        return {}

    builder.add_node("memory_load", memory_load_node)
    builder.add_node("planner", planner_node)
    builder.add_node("action", action_node)
    builder.add_node("thought", thought_node)
    builder.add_node("update_step", update_step_node)
    builder.add_node("final_answer", final_answer_node)
    builder.add_node("memory_persist", memory_persist_node)

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "planner")
    builder.add_edge("planner", "action")
    builder.add_edge("action", "thought")
    builder.add_edge("thought", "update_step")

    builder.add_conditional_edges(
        "update_step",
        router,
        {"action": "action", "final_answer": "final_answer"},
    )

    builder.add_edge("final_answer", "memory_persist")
    builder.add_edge("memory_persist", END)

    return builder.compile(checkpointer=MemorySaver(), store=store), store
## Run the Agent
graph, store = build_graph_with_memory()

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

user_query = input("Ask your question: ")

initial_state: OrchestrationState = {
    "messages": [{"role": "user", "content": user_query}],
    "plan": None,
    "current_step": 0,
    "observations": [],
    "thought": None,
    "final_answer": None,
    "memory": None,
    "memory_update_text": None,
    "user_id": "user_1",
}

print("\n" + "="*60)
print("EXECUTING AGENT...")
print("="*60)

result = graph.invoke(initial_state, config)

print("\n" + "="*60)
print("FINAL ANSWER")
print("="*60)
print(result["final_answer"])

print("\n" + "="*60)
print("MEMORY CONTENTS (After Run)")
print("="*60)
memory_content = get_long_term_memory(store, "user_1")
if memory_content:
    print(memory_content)
else:
    print("No memory stored yet.")
print("="*60)

print("\n" + "="*60)
print("NEW INFORMATION LEARNED (This Session)")
print("="*60)
if result.get("memory_update_text"):
    print(result["memory_update_text"])
else:
    print("No new memory updates.")
print("="*60)
## Utility: Inspect Memory Function
def inspect_memory(store: InMemoryStore, user_id: str):
    """
    Inspect and print detailed memory contents for a specific user
    """
    print(f"\n{'='*60}")
    print(f"Memory Inspection for User: {user_id}")
    print(f"{'='*60}")
    
    namespace = ("user_memory",)
    try:
        item = store.get(namespace, user_id)
        
        if item and hasattr(item, 'value'):
            memory_value = item.value
            print(f"\nMemory exists: Yes")
            print(f"Memory type: {type(memory_value)}")
            print(f"Memory length: {len(str(memory_value))} characters")
            print(f"\nMemory content:")
            print("-" * 60)
            print(memory_value)
            print("-" * 60)
        else:
            print("\nMemory exists: No")
            print("This user has no stored memory yet.")
            
    except Exception as e:
        print(f"\nMemory exists: No")
        print(f"Error: {e}")
    
    print(f"{'='*60}\n")

# Usage example:

inspect_memory(store, "user_1")
# That the user can ask followup next to one questions to other and try to reduce the latency