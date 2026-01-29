"""

Architecture:
User Input → Load Memory → Action → Final Answer → Memory Update → Feedback → END
"""


import os
import uuid
from typing import TypedDict, Annotated, Dict, Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient
from langsmith import get_current_run_tree
from huggingface_hub import InferenceClient
from ddgs import DDGS


# =========================
# ENV
# =========================

assert os.getenv("HF_TOKEN")
assert os.getenv("LANGCHAIN_PROJECT")
assert os.getenv("LANGCHAIN_API_KEY")

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))
langsmith_client = LangSmithClient()

print("LangSmith project:", PROJECT_NAME)

PROFILE_NS = ("user_profile",)
CONVO_NS = ("user_conversation",)



# =========================
# LLM WRAPPER (WITH RUN ID)
# =========================

def call_llm(prompt: str, *, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512):
    with trace(
        name="llm_call",
        project_name=PROJECT_NAME,
        metadata={"model": model},
        tags=["agent"],
    ) as run:
        resp = hf_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip(), run.id


# =========================
# WEB SEARCH
# =========================

def web_search(query: str, k: int = 2) -> str:
    with trace(
        name="web_search",
        project_name=PROJECT_NAME,
        metadata={"query": query},
        tags=["tool"],
    ):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k):
                results.append(
                    f"{r.get('title','')}\n{r.get('body','')}\nSource: {r.get('href','')}"
                )
        return "\n\n".join(results) if results else "No results found."


# =========================
# STATE
# =========================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    profile_memory: Optional[str]
    conversation_memory: Optional[str]
    observation: Optional[str]
    final_answer: Optional[str]
    profile_update: Optional[str]
    conversation_update: Optional[str]
    user_id: str
    last_run_id: Optional[str]


# =========================
# MEMORY HELPERS
# =========================

def get_memory(store: InMemoryStore, namespace: tuple, user_id: str) -> str:
    try:
        item = store.get(namespace, user_id)
        if item is None:
            return ""
        return item.value if hasattr(item, "value") else item
    except Exception:
        return ""

def update_memory(store: InMemoryStore, namespace: tuple, user_id: str, new_value: str, existing: str):
    if not new_value or new_value.strip().lower() == "none":
        return
    combined = (existing + "\n" + new_value).strip() if existing else new_value
    store.put(namespace, user_id, combined)


# =========================
# NODES
# =========================

def memory_load_node(state: AgentState, store: InMemoryStore):
    return {
        "profile_memory": get_memory(store, PROFILE_NS, state["user_id"]),
        "conversation_memory": get_memory(store, CONVO_NS, state["user_id"]),
    }

def action_node(state: AgentState):
    last = state["messages"][-1]
    user_query = last.content

    prompt = f"""
User profile memory:
{state.get("profile_memory") or "None"}

Conversation memory:
{state.get("conversation_memory") or "None"}

User question:
{user_query}

If you cannot answer confidently, respond ONLY: INSUFFICIENT
"""

    llm_answer, run_id = call_llm(prompt)

    if "insufficient" in llm_answer.lower():
        observation = web_search(user_query)
    else:
        observation = llm_answer

    return {"observation": observation, "last_run_id": run_id}

def final_answer_node(state: AgentState):
    last = state["messages"][-1]
    user_query = last.content

    answer_prompt = f"""
User memory:
{state.get("profile_memory")}

Conversation memory:
{state.get("conversation_memory")}

Question:
{user_query}

Info:
{state.get("observation")}

Answer clearly:
"""

    answer, run_id = call_llm(answer_prompt)

    profile_prompt = f"""
Extract ONLY user facts (name, preferences).
Return NONE if nothing.

User message:
{user_query}
"""

    profile_update, _ = call_llm(profile_prompt)

    convo_prompt = f"""
Summarize the interaction in one sentence.
Format: User asked about ...

User: {user_query}
Assistant: {answer}
"""

    conversation_update, _ = call_llm(convo_prompt)

    return {
        "final_answer": answer,
        "profile_update": profile_update,
        "conversation_update": conversation_update,
        "last_run_id": run_id,
    }

def memory_persist_node(state: AgentState, store: InMemoryStore):
    update_memory(store, PROFILE_NS, state["user_id"], state.get("profile_update"), state.get("profile_memory", ""))
    update_memory(store, CONVO_NS, state["user_id"], state.get("conversation_update"), state.get("conversation_memory", ""))
    return {}
# =========================
# GRAPH
# =========================

def build_graph():
    store = InMemoryStore()
    builder = StateGraph(AgentState)

    builder.add_node("memory_load", lambda s: memory_load_node(s, store))
    builder.add_node("action", action_node)
    builder.add_node("final_answer", final_answer_node)
    builder.add_node("memory_persist", lambda s: memory_persist_node(s, store))

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "action")
    builder.add_edge("action", "final_answer")
    builder.add_edge("final_answer", "memory_persist")
    builder.add_edge("memory_persist", END)

    return builder.compile(checkpointer=MemorySaver(), store=store), store

from langsmith.run_helpers import trace 
from langsmith import  Client as LangSmithClient
import uuid
def run_agent(query: str, graph, store, thread_id: str, user_id: str):
    """
    Runs the agent, logs to LangSmith, and records user feedback
    into BOTH:
      - Run feedback (Runs tab)
      - Thread metadata (Threads tab)
    """

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
            # use uuid for user identification
            "user_id": uuid.uuid1()
        }
    }

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "profile_memory": None,
        "conversation_memory": None,
        "observation": None,
        "final_answer": None,
        "profile_update": None,
        "conversation_update": None,
        "user_id": user_id,
    }

    # -----------------------------
    # TRACE THE FULL AGENT RUN
    # -----------------------------
    with trace(
        name="agent_run",
        project_name=PROJECT_NAME,
        inputs={"question": query},
        tags=["agent"],
    ) as run:

        result = graph.invoke(initial_state, config)

        final_answer = result["final_answer"]

        print("\nAI:", final_answer)

        # -----------------------------
        # ASK USER FOR FEEDBACK
        # -----------------------------
        rating = input("\nRate this answer (1–5) or press Enter to skip: ").strip()

        if rating:
            try:
                rating_value = int(rating)
                if 1 <= rating_value <= 5:
                    # -----------------------------
                    # SAVE FEEDBACK ON RUN
                    # -----------------------------
                    langsmith_client.create_feedback(
                        run.id,
                        key="user_rating_feedback",
                        score=rating_value,
                        comment=f"User rated response {rating_value}/5",
                    )

                    # -----------------------------
                    # SAVE FEEDBACK IN THREAD METADATA
                    # -----------------------------
                    langsmith_client.update_run(
                        run.id,
                        extra={
                            "thread_feedback": rating_value,
                            "user_id": user_id
                        }
                    )

                    print(" Feedback recorded.")

                else:
                    print(" Rating must be between 1 and 5.")
            except ValueError:
                print(" Invalid rating, skipped.")

    # -----------------------------
    # SHOW MEMORY
    # -----------------------------
    print("\n[PROFILE MEMORY]")
    print(get_memory(store, PROFILE_NS, user_id) or "None")

    print("\n[CONVERSATION MEMORY]")
    print(get_memory(store, CONVO_NS, user_id) or "None")

    return result



# =========================
# MAIN
# =========================

if __name__ == "__main__":
    graph, store = build_graph()
    thread_id = str(uuid.uuid4())
    user_id = "user_1"

    print("DUAL MEMORY AGENT WITH FEEDBACK")
    print("Type quit to exit")

    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "quit":
            break
        run_agent(q, graph, store, thread_id, user_id)
