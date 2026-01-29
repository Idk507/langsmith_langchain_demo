"""
Architecture with Intelligent Caching:
User Input → Load Memory → Check Cache → Action (if needed) → Final Answer → Memory Update → Feedback → END
"""

import os
import uuid
import json
import hashlib
from typing import TypedDict, Annotated, Dict, Any, Optional
from difflib import SequenceMatcher
import time 
import numpy as np
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

from sentence_transformers import SentenceTransformer

# Caching imports
import shelve
from diskcache import Cache as DiskCache



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
CACHE_TTL = 3600
SHELVE_PATH = "./query_cache.db"

disk_cache = DiskCache("./cache_dir")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compute_query_hash(query: str, user_id: str) -> str:
    key = f"{query.lower().strip()}|{user_id}"
    return hashlib.md5(key.encode()).hexdigest()

def semantic_cache_lookup(query: str, user_id: str, similarity_threshold: float = 0.80):
    """
    Hybrid cache:
      1. Exact hash match (DiskCache)
      2. Semantic similarity match (Shelve + embeddings)
    """

    with trace(
        name="cache_lookup",
        project_name=PROJECT_NAME,
        metadata={"query": query, "user_id": user_id},
        tags=["cache"],
    ) as run:

        try:
            # ---- FAST EXACT MATCH ----
            cache_key = compute_query_hash(query, user_id)
            if cache_key in disk_cache:
                run.end(outputs={"hit": True, "mode": "exact"})
                return disk_cache[cache_key]

            # ---- SEMANTIC MATCH ----
            query_embedding = embedding_model.encode(query)

            with shelve.open(SHELVE_PATH) as db:
                for k in db.keys():
                    cached = db[k]

                    if cached["user_id"] != user_id:
                        continue

                    cached_emb = np.array(cached["embedding"])
                    sim = cosine_similarity(query_embedding, cached_emb)

                    if sim >= similarity_threshold:
                        run.end(
                            outputs={
                                "hit": True,
                                "mode": "semantic",
                                "similarity": sim,
                                "matched_query": cached["query"],
                            }
                        )
                        return cached

            run.end(outputs={"hit": False, "mode": "miss"})
            return None

        except Exception as e:
            run.end(outputs={"hit": False, "error": str(e)})
            return None


def semantic_cache_write(query: str, user_id: str, final_answer: str):
    with trace(
        name="cache_write",
        project_name=PROJECT_NAME,
        metadata={"query": query, "user_id": user_id},
        tags=["cache"],
    ) as run:

        try:
            embedding = embedding_model.encode(query).tolist()
            cache_key = compute_query_hash(query, user_id)

            cache_data = {
                "query": query,
                "embedding": embedding,
                "user_id": user_id,
                "final_answer": final_answer,
                "timestamp": time.time(),
            }

            with shelve.open(SHELVE_PATH) as db:
                db[cache_key] = cache_data

            disk_cache.set(cache_key, cache_data, expire=CACHE_TTL)

            run.end(outputs={"status": "stored", "cache_key": cache_key})

        except Exception as e:
            run.end(outputs={"status": "error", "error": str(e)})


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
    cache_hit: Optional[bool]  # Track if response came from cache


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
# NODES (WITH CACHING)
# =========================

def memory_load_node(state: AgentState, store: InMemoryStore):
    return {
        "profile_memory": get_memory(store, PROFILE_NS, state["user_id"]),
        "conversation_memory": get_memory(store, CONVO_NS, state["user_id"]),
        "cache_hit": False
    }

def action_node(state: AgentState):
    """
    Action node with semantic cache-first strategy.
    Always tries cache before calling LLM.
    """

    last = state["messages"][-1]
    user_query = last.content

    user_id = state["user_id"]
    profile_mem = state.get("profile_memory") or ""
    convo_mem = state.get("conversation_memory") or ""

    # =========================
    # 1. SEMANTIC CACHE LOOKUP
    # =========================
    cached = semantic_cache_lookup(user_query, user_id)

    if cached:
        return {
            "observation": cached.get("final_answer"),
            "final_answer": cached.get("final_answer"),
            "cache_hit": True,
            "last_run_id": None,
        }

    # =========================
    # 2. CACHE MISS → CALL LLM
    # =========================
    prompt = f"""
User profile memory:
{profile_mem or "None"}

Conversation memory:
{convo_mem or "None"}

User question:
{user_query}

If you cannot answer confidently, respond ONLY: INSUFFICIENT
"""

    llm_answer, run_id = call_llm(prompt)

    # =========================
    # 3. TOOL FALLBACK (OPTIONAL)
    # =========================
    if "insufficient" in llm_answer.lower():
        observation = web_search(user_query)
    else:
        observation = llm_answer

    return {
        "observation": observation,
        "cache_hit": False,
        "last_run_id": run_id,
    }


def final_answer_node(state: AgentState):
    """
    Generate final answer.
    If cache hit, skip LLM calls for memory updates.
    """
    last = state["messages"][-1]
    user_query = last.content
    
    # If cache hit, return cached answer
    if state.get("cache_hit"):
        return {
            "final_answer": state.get("final_answer"),
            "profile_update": None,
            "conversation_update": None,
            "last_run_id": None
        }
    
    # Normal flow - generate answer
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
    
    # ===== SAVE TO CACHE =====
    save_to_cache(
        user_query,
        state["user_id"],
        state.get("profile_memory") or "",
        state.get("conversation_memory") or "",
        state.get("observation"),
        answer
    )

    return {
        "final_answer": answer,
        "profile_update": profile_update,
        "conversation_update": conversation_update,
        "last_run_id": run_id,
    }


def memory_persist_node(state: AgentState, store: InMemoryStore):
    """
    Only update memory if not a cache hit.
    """
    if state.get("cache_hit"):
        return {}
    
    update_memory(
        store, PROFILE_NS, state["user_id"], 
        state.get("profile_update"), 
        state.get("profile_memory", "")
    )
    update_memory(
        store, CONVO_NS, state["user_id"], 
        state.get("conversation_update"), 
        state.get("conversation_memory", "")
    )
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



def run_agent(query: str, graph, store, thread_id: str, user_id: str):
    """
    Runs the agent with caching support.
    """

    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {
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
        "cache_hit": False
    }

    with trace(
        name="agent_run",
        project_name=PROJECT_NAME,
        inputs={"question": query},
        tags=["agent"],
    ) as run:

        result = graph.invoke(initial_state, config)

        final_answer = result["final_answer"]
        cache_hit = result.get("cache_hit", False)

        print("\nAI:", final_answer)
        
        if cache_hit:
            print("✓ [Response served from cache]")

        # Feedback collection
        rating = input("\nRate this answer (1–5) or press Enter to skip: ").strip()

        if rating:
            try:
                rating_value = int(rating)
                if 1 <= rating_value <= 5:
                    langsmith_client.create_feedback(
                        run.id,
                        key="user_rating_feedback",
                        score=rating_value,
                        comment=f"User rated response {rating_value}/5 (cached: {cache_hit})",
                    )

                    langsmith_client.update_run(
                        run.id,
                        extra={
                            "thread_feedback": rating_value,
                            "user_id": user_id,
                            "cache_hit": cache_hit
                        }
                    )

                    print("✓ Feedback recorded.")

                else:
                    print("⚠ Rating must be between 1 and 5.")
            except ValueError:
                print("⚠ Invalid rating, skipped.")

    # Show memory
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

    print("=" * 60)
    print("DUAL MEMORY AGENT WITH INTELLIGENT CACHING")
    print("=" * 60)
    print("Commands:")
    print("  - 'quit' to exit")
    print("  - 'clear cache' to clear all cached responses")
    print("=" * 60)

    while True:
        q = input("\nYour question: ").strip()
        
        if q.lower() == "quit":
            break
        
        if q.lower() == "clear cache":
            clear_cache()
            continue
            
        if not q:
            continue
            
        run_agent(q, graph, store, thread_id, user_id)


