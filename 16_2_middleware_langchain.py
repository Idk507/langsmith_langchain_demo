# persistent_agent_with_middleware.py
"""
Persistent Memory Agent with Middleware Manager
- Integrates middleware hooks (before_model, modify_model_request, after_model)
- Includes built-in middleware implementations (Summarization, HumanInTheLoop,
  ModelFallback, PIIMiddleware, ModelCallLimit)
- Uses a PersistentMemoryStore (shelve-based) and semantic/disk cache (diskcache + shelve)
- Integrates middleware into call_llm() and web_search()
- Keeps LangSmith tracing and Hugging Face InferenceClient model usage
Dependencies (pip):
    pip install huggingface-hub ddgs diskcache sentence-transformers langsmith langgraph langchain
    # langchain & langgraph versions should be recent pre-release compatible
"""
from __future__ import annotations
import os
import uuid
import json
import time
import shelve
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple

import numpy as np

try:
    from diskcache import Cache as DiskCache
except Exception:
    DiskCache = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from huggingface_hub import InferenceClient

# LangSmith tracing (keeps your observability)
from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient

# LangGraph/StateGraph pieces (you already use these)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage

# Try to import LangChain middleware types (if available in your environment).
# If not available, we use a small internal base class that mirrors the hooks.
try:
    from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
    LANGCHAIN_MIDDLEWARE_AVAILABLE = True
except Exception:
    LANGCHAIN_MIDDLEWARE_AVAILABLE = False

    class ModelRequest(dict):
        """
        Lightweight fallback for ModelRequest:
         - model: str (model id) or any
         - messages: list
         - max_tokens: int
         - metadata: dict
         - tools: list
         - response_format: optional
        """
        pass

    class ModelResponse(dict):
        """Fallback wrapper for model responses"""
        pass

    class AgentMiddleware:
        """Simple fallback base class mirroring the hook signatures used in docs."""
        def before_model(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            return None

        def modify_model_request(self, request: ModelRequest, state: Dict[str, Any]) -> ModelRequest:
            return request

        def after_model(self, state: Dict[str, Any], response: ModelResponse) -> Optional[Dict[str, Any]]:
            return None

# -------------------------
# Configuration / Clients
# -------------------------
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "default-project")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set in environment")

hf_client = InferenceClient(token=HF_TOKEN)
langsmith_client = LangSmithClient()  # optional usage

# Caches and embedding model (if available)
DISK_CACHE_DIR = os.getenv("KV_DISKCACHE_DIR", "./cache_dir")
SHELVE_PATH = os.getenv("KV_SHELVE_PATH", "./kv_query_cache.db")
MEMORY_PATH = os.getenv("MEMORY_PATH", "./persistent_memory.db")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

disk_cache = DiskCache(DISK_CACHE_DIR) if DiskCache is not None else None
embedding_model = None
if SentenceTransformer is not None:
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        embedding_model = None

# -------------------------
# Persistent memory store
# -------------------------
class PersistentMemoryStore:
    """Shelve-based persistent memory for profiles and conversation notes."""
    def __init__(self, db_path: str = MEMORY_PATH):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        try:
            with shelve.open(self.db_path) as db:
                pass
        except Exception as e:
            print("Warning: cannot create persistent memory file:", e)

    def _make_key(self, namespace: tuple, user_id: str) -> str:
        return f"{':'.join(namespace)}|{user_id}"

    def get(self, namespace: tuple, user_id: str) -> str:
        try:
            with shelve.open(self.db_path) as db:
                key = self._make_key(namespace, user_id)
                return db.get(key, "")
        except Exception as e:
            print("Memory read error:", e)
            return ""

    def put(self, namespace: tuple, user_id: str, value: str):
        try:
            with shelve.open(self.db_path) as db:
                key = self._make_key(namespace, user_id)
                db[key] = value
        except Exception as e:
            print("Memory write error:", e)

    def delete(self, namespace: tuple, user_id: str):
        try:
            with shelve.open(self.db_path) as db:
                key = self._make_key(namespace, user_id)
                if key in db:
                    del db[key]
        except Exception as e:
            print("Memory delete error:", e)

    def list_all_users(self) -> List[str]:
        users = set()
        try:
            with shelve.open(self.db_path) as db:
                for k in db.keys():
                    if '|' in k:
                        users.add(k.split('|')[1])
        except Exception:
            pass
        return list(users)

    def clear_user_memory(self, user_id: str):
        try:
            with shelve.open(self.db_path) as db:
                keys = [k for k in db.keys() if k.endswith(f"|{user_id}")]
                for k in keys:
                    del db[k]
        except Exception as e:
            print("Error clearing user memory:", e)

# -------------------------
# Semantic + disk cache helpers
# -------------------------
def compute_query_hash(query: str, user_id: str) -> str:
    key = f"{query.lower().strip()}|{user_id}"
    return hashlib.md5(key.encode()).hexdigest()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def semantic_cache_lookup(query: str, user_id: str, similarity_threshold: float = 0.80):
    """Exact disk cache lookup then optional semantic search via stored embeddings."""
    try:
        cache_key = compute_query_hash(query, user_id)
        if disk_cache is not None and cache_key in disk_cache:
            return disk_cache[cache_key]

        if embedding_model is None:
            return None

        q_emb = embedding_model.encode(query)
        with shelve.open(SHELVE_PATH) as db:
            for k in db.keys():
                item = db[k]
                if item.get("user_id") != user_id:
                    continue
                cached_emb = np.array(item["embedding"])
                sim = cosine_similarity(q_emb, cached_emb)
                if sim >= similarity_threshold:
                    return item
        return None
    except Exception as e:
        print("Cache lookup error:", e)
        return None

def semantic_cache_write(query: str, user_id: str, final_answer: str):
    try:
        emb = embedding_model.encode(query).tolist() if embedding_model is not None else None
        cache_key = compute_query_hash(query, user_id)
        data = {"query": query, "embedding": emb, "user_id": user_id, "final_answer": final_answer, "timestamp": time.time()}
        if disk_cache is not None:
            disk_cache.set(cache_key, data, expire=CACHE_TTL)
        with shelve.open(SHELVE_PATH) as db:
            db[cache_key] = data
    except Exception as e:
        print("Cache write error:", e)

def clear_cache():
    try:
        if disk_cache is not None:
            disk_cache.clear()
        with shelve.open(SHELVE_PATH) as db:
            db.clear()
    except Exception as e:
        print("Clear cache error:", e)

# -------------------------
# Middleware manager
# -------------------------
class MiddlewareManager:
    """Run middleware hooks in the order expected by LangChain docs:
       - before_model (in order)
       - modify_model_request (in order)
       - after_model (in reverse order)
    """
    def __init__(self, middlewares: Optional[List[AgentMiddleware]] = None):
        self.middlewares = middlewares or []

    def add(self, mw: AgentMiddleware):
        self.middlewares.append(mw)

    def run_before_model(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply before_model hooks in order; merge returned patches."""
        patch = {}
        for mw in self.middlewares:
            try:
                p = mw.before_model(state)
                if p:
                    patch.update(p)
            except Exception as e:
                print(f"Middleware before_model error ({mw.__class__.__name__}):", e)
        return patch

    def run_modify_model_request(self, request: ModelRequest, state: Dict[str, Any]) -> ModelRequest:
        req = request
        for mw in self.middlewares:
            try:
                req = mw.modify_model_request(req, state)
            except Exception as e:
                print(f"Middleware modify_model_request error ({mw.__class__.__name__}):", e)
        return req

    def run_after_model(self, state: Dict[str, Any], response: ModelResponse) -> Dict[str, Any]:
        """Run after_model in reverse order and collect patches"""
        patch = {}
        for mw in reversed(self.middlewares):
            try:
                p = mw.after_model(state, response)
                if p:
                    patch.update(p)
            except Exception as e:
                print(f"Middleware after_model error ({mw.__class__.__name__}):", e)
        return patch

# -------------------------
# Built-in middleware implementations (practical versions)
# -------------------------
class SummarizationMiddleware(AgentMiddleware):
    """Summarizes message history when tokens exceed a threshold.
    Uses the same LLM wrapper but with small max_new_tokens; returns a 'messages' patch.
    """
    def __init__(self, summarize_prompt: str = "Summarize earlier context concisely:",
                 max_tokens_before_summary: int = 4000, messages_to_keep: int = 20):
        self.summarize_prompt = summarize_prompt
        self.max_tokens_before_summary = max_tokens_before_summary
        self.messages_to_keep = messages_to_keep

    def before_model(self, state):
        messages = state.get("messages", [])
        # rough token heuristic = character count / 4 (approx). This is simple and safe.
        total_chars = sum(len(m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) for m in messages)
        if total_chars > self.max_tokens_before_summary:
            # summarize earlier context
            context_to_summarize = messages[:-self.messages_to_keep]
            combined = "\n\n".join(
                (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) for m in context_to_summarize
            )
            # call model to summarize quickly (synchronous)
            prompt = f"{self.summarize_prompt}\n\nContext:\n{combined}\n\nSummary:"
            # Note: We call hf_client directly to avoid recursion through call_llm (which itself calls middleware).
            try:
                with trace(name="middleware_summarize", project_name=PROJECT_NAME):
                    resp = hf_client.chat_completion(
                        model="Qwen/Qwen2.5-72B-Instruct",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=256,
                    )
                    summary = resp.choices[0].message.content.strip()
                    # Return messages patch: keep summary + last n messages
                    patched_messages = [{"role": "system", "content": f"Summary of earlier context: {summary}"}]
                    patched_messages += messages[-self.messages_to_keep:]
                    return {"messages": patched_messages}
            except Exception as e:
                print("Summarization failure:", e)
        return None

class SimpleHumanInTheLoopMiddleware(AgentMiddleware):
    """Synchronous HITL: prompts the operator on console when a tool call or sensitive action is detected.
       For durable, resumable interrupts you should replace the input() with langgraph.types.interrupt
       inside the graph node (requires a checkpointer like MemorySaver()).
    """
    def __init__(self, require_approval_for_tools: Optional[List[str]] = None, message_prefix: str = "Approval required:"):
        self.require_approval_for_tools = set(require_approval_for_tools or [])
        self.message_prefix = message_prefix

    def after_model(self, state, response):
        # Detect candidate tool calls or anything in the last assistant output indicating a "tool"
        last_msg = state.get("messages", [])[-1] if state.get("messages") else {}
        content = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", "")
        # Basic detection: presence of "CALL_TOOL:" or explicit phrase - adapt per your agent design
        for tool in self.require_approval_for_tools:
            if tool in content:
                # ask for approval synchronously
                print(f"\n{self.message_prefix} The agent wants to call tool: {tool}")
                print("Tool payload excerpt:", (content[:400] + "...") if content else "None")
                resp = input("Type YES to approve, anything else to deny: ").strip().lower()
                if resp == "yes":
                    return None
                else:
                    # modify state to replace last message with an abort note
                    abort_msg = {"role": "assistant", "content": f"Tool call {tool} aborted by human."}
                    msgs = state.get("messages", [])[:-1] + [abort_msg]
                    return {"messages": msgs, "jump_to": "__end__"}
        return None

class PIIMiddleware(AgentMiddleware):
    """Detects simple PII (phone/email/SSN-ish) and redacts it in before_model."""
    EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_RE = re.compile(r"(?:\+?\d{1,3})?[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{2,4}")

    def before_model(self, state):
        patched = False
        messages = state.get("messages", [])
        new_messages = []
        for m in messages:
            text = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            # redact email & phone
            redacted = PIIMiddleware.EMAIL_RE.sub("[REDACTED_EMAIL]", text)
            redacted = PIIMiddleware.PHONE_RE.sub("[REDACTED_PHONE]", redacted)
            if redacted != text:
                patched = True
            if isinstance(m, dict):
                new_messages.append({"role": m.get("role", "user"), "content": redacted})
            else:
                # preserve object type minimally: convert to dict
                new_messages.append({"role": getattr(m, "role", "user"), "content": redacted})
        if patched:
            return {"messages": new_messages}
        return None

class ModelFallbackMiddleware(AgentMiddleware):
    """If the primary model fails, suggest fallback model names. This middleware
    modifies the ModelRequest.model to attempt alternates (string-based).
    """
    def __init__(self, fallback_models: Optional[List[str]] = None):
        self.fallback_models = fallback_models or []

    def modify_model_request(self, request, state):
        # request is dict-like; ensure it contains model key
        if not request.get("model"):
            request["model"] = "Qwen/Qwen2.5-72B-Instruct"
        # Attach fallback list into metadata for our call_llm to use on failure
        meta = request.get("metadata", {})
        meta["_fallback_models"] = self.fallback_models
        request["metadata"] = meta
        return request

class ModelCallLimitMiddleware(AgentMiddleware):
    """Track count of model calls per run in provided state and stop if exceeded.
       This is a conservative, pre-check in before_model.
    """
    def __init__(self, max_calls_per_run: int = 6):
        self.max_calls = max_calls_per_run

    def before_model(self, state):
        counters = state.setdefault("_meta", {})
        calls = counters.get("model_calls", 0)
        if calls >= self.max_calls:
            # signal an early stop via jump_to
            return {"jump_to": "__end__", "final_answer": "Model call limit reached for this run."}
        # otherwise nothing — increment in modify_model_request to ensure it's tied to a model call
        return None

    def modify_model_request(self, request, state):
        counters = state.setdefault("_meta", {})
        counters["model_calls"] = counters.get("model_calls", 0) + 1
        return request

# -------------------------
# Agent core: call_llm and web_search wired to middleware
# -------------------------
middleware_manager = MiddlewareManager([
    SummarizationMiddleware(),
    PIIMiddleware(),
    ModelCallLimitMiddleware(max_calls_per_run=8),
    ModelFallbackMiddleware(fallback_models=["zai-org/GLM-4.7-Flash", "Qwen/Qwen2.5-72B-Instruct"]),
    SimpleHumanInTheLoopMiddleware(require_approval_for_tools=["delete_database", "send_email"]),
    # Add more middleware instances here...
])

def call_llm_with_middleware(state: Dict[str, Any], prompt: str, *, model: str = "Qwen/Qwen2.5-72B-Instruct",
                             max_new_tokens: int = 512) -> Tuple[str, Optional[str]]:
    """
    Central LLM call that passes through middleware hooks.
    - state: current agent state dictionary (mutable)
    - returns (assistant_text, run_id)
    """
    # 1) before_model hooks
    try:
        patch = middleware_manager.run_before_model(state)
        if patch:
            state.update(patch)
            # honor jump_to early end
            if patch.get("jump_to") == "__end__":
                # return early final_answer if supplied
                return (patch.get("final_answer", "Execution stopped by middleware."), None)
    except Exception as e:
        print("before_model middleware failed:", e)

    # 2) build ModelRequest
    req = ModelRequest()
    req["model"] = model
    req["messages"] = state.get("messages", [{"role": "user", "content": prompt}])
    req["max_tokens"] = max_new_tokens
    req["metadata"] = {"prompt_excerpt": (prompt[:400] + "...") if prompt else ""}
    # allow middleware to modify request
    req = middleware_manager.run_modify_model_request(req, state)

    # 3) Call model with robust fallback behavior
    # model may be a string or provider object — we assume string model ids for HF InferenceClient.
    attempt_models = [req.get("model")]
    if isinstance(req.get("metadata", {}), dict):
        fallbacks = req["metadata"].get("_fallback_models") or []
        for fb in fallbacks:
            if fb not in attempt_models:
                attempt_models.append(fb)

    last_exc = None
    for attempt_model in attempt_models:
        try:
            with trace(name="llm_call", project_name=PROJECT_NAME, metadata={"model": attempt_model}):
                resp = hf_client.chat_completion(
                    model=attempt_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_new_tokens,
                )
                # normalize
                assistant_text = ""
                try:
                    assistant_text = resp.choices[0].message.content or getattr(resp.choices[0].message, "reasoning_content", "")
                except Exception:
                    assistant_text = str(resp)
                # build ModelResponse-like dict for middleware
                response_obj = ModelResponse({"assistant_text": assistant_text, "usage": getattr(resp, "usage", None)})
                # apply after_model hooks (reverse order)
                patch_after = middleware_manager.run_after_model(state, response_obj)
                if patch_after:
                    state.update(patch_after)
                return assistant_text, getattr(resp, "id", None)
        except Exception as e:
            last_exc = e
            # try next fallback model
            print(f"Model attempt failed for {attempt_model}: {e}")
            continue
    # all attempts failed
    print("All model attempts failed; raising last exception.")
    raise RuntimeError(f"LLM calls failed: {last_exc}")

def web_search_with_middleware(state: Dict[str, Any], query: str, k: int = 3) -> str:
    """Wrap web_search to pass through before_model and after_model hooks as well."""
    # We can let before_model mutate state (summarization/PII)
    _ = middleware_manager.run_before_model(state)
    try:
        with trace(name="web_search", project_name=PROJECT_NAME, metadata={"query": query}):
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    t = r.get("title", "")
                    b = r.get("body", "")
                    href = r.get("href", "")
                    results.append(f"{t}\n{b}\nSource: {href}")
            output = "\n\n".join(results) if results else "No results found."
    except Exception as e:
        output = f"Search failed: {e}"
    # treat web search result as a ModelResponse for after_model processing
    resp = ModelResponse({"assistant_text": output})
    patch_after = middleware_manager.run_after_model(state, resp)
    if patch_after:
        state.update(patch_after)
    return output

# -------------------------
# Integration: Persistent agent graph with middleware-aware nodes
# -------------------------
PROFILE_NS = ("user_profile",)
CONVO_NS = ("user_conversation",)

def get_memory(pm_store: PersistentMemoryStore, namespace: tuple, user_id: str) -> str:
    return pm_store.get(namespace, user_id) or ""

def update_memory(pm_store: PersistentMemoryStore, namespace: tuple, user_id: str, new_value: str, existing: str):
    if not new_value or new_value.strip().lower() == "none":
        return
    combined = (existing + "\n" + new_value).strip() if existing else new_value
    pm_store.put(namespace, user_id, combined)

# Build graph nodes that use call_llm_with_middleware & web_search_with_middleware
def build_persistent_agent_graph():
    store = PersistentMemoryStore()
    builder = StateGraph(dict)  # dynamic TypedDict not required here for flexibility

    # memory load node
    def memory_load(state: dict):
        user_id = state.get("user_id", "user_1")
        return {
            "profile_memory": get_memory(store, PROFILE_NS, user_id),
            "conversation_memory": get_memory(store, CONVO_NS, user_id),
            "cache_hit": False
        }

    def action_node(state: dict):
        # similar to your earlier design: check cache then call LLM or web search via middleware
        last = state["messages"][-1]
        user_query = last["content"] if isinstance(last, dict) else getattr(last, "content", "")
        user_id = state.get("user_id", "user_1")

        # cache-first
        cached = semantic_cache_lookup(user_query, user_id)
        if cached:
            return {"observation": cached["final_answer"], "final_answer": cached["final_answer"], "cache_hit": True}

        # build a context-respecting prompt
        profile_mem = state.get("profile_memory", "")
        convo_mem = state.get("conversation_memory", "")

        prompt = f"""
User profile memory:
{profile_mem or 'None'}

Conversation memory:
{convo_mem or 'None'}

User question:
{user_query}

If you cannot answer confidently, respond ONLY: INSUFFICIENT
"""
        # ensure messages in state for middleware to inspect
        state.setdefault("messages", [{"role": "user", "content": user_query}])
        # call model via middleware
        assistant_text, run_id = call_llm_with_middleware(state, prompt, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512)
        if "insufficient" in (assistant_text or "").lower():
            observation = web_search_with_middleware(state, user_query, k=2)
        else:
            observation = assistant_text
        return {"observation": observation, "cache_hit": False, "last_run_id": run_id}

    def final_answer_node(state: dict):
        # if cache hit, skip heavy steps
        if state.get("cache_hit"):
            return {"final_answer": state.get("final_answer"), "profile_update": None, "conversation_update": None}

        user_query = state["messages"][-1]["content"] if isinstance(state["messages"][-1], dict) else getattr(state["messages"][-1], "content", "")
        all_observations = state.get("observation", "")
        profile_mem = state.get("profile_memory", "")

        answer_prompt = f"""
User memory:
{profile_mem}

Question:
{user_query}

Info:
{all_observations}

Answer clearly:
"""
        answer_text, run_id = call_llm_with_middleware(state, answer_prompt, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=512)

        # extract profile facts (small prompt)
        profile_prompt = f"""
Extract ONLY explicit user facts (name, preferences). Return NONE if nothing.

User message:
{user_query}
"""
        profile_update, _ = call_llm_with_middleware(state, profile_prompt, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=128)

        convo_prompt = f"""Summarize the interaction in one sentence.
User: {user_query}
Assistant: {answer_text}
"""
        conversation_update, _ = call_llm_with_middleware(state, convo_prompt, model="Qwen/Qwen2.5-72B-Instruct", max_new_tokens=128)

        # store to semantic cache to speed later
        semantic_cache_write(user_query, state.get("user_id", "user_1"), answer_text)

        return {"final_answer": answer_text, "profile_update": profile_update, "conversation_update": conversation_update, "last_run_id": run_id}

    def memory_persist_node(state: dict):
        # persist only if not cache_hit
        if state.get("cache_hit"):
            return {}
        user_id = state.get("user_id", "user_1")
        update_memory(store, PROFILE_NS, user_id, state.get("profile_update"), state.get("profile_memory", ""))
        update_memory(store, CONVO_NS, user_id, state.get("conversation_update"), state.get("conversation_memory", ""))
        return {}

    builder.add_node("memory_load", memory_load)
    builder.add_node("action", action_node)
    builder.add_node("final_answer", final_answer_node)
    builder.add_node("memory_persist", memory_persist_node)

    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "action")
    builder.add_edge("action", "final_answer")
    builder.add_edge("final_answer", "memory_persist")
    builder.add_edge("memory_persist", END)

    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    return graph, store

# -------------------------
# Simple CLI runner
# -------------------------
def run_agent_cli():
    graph, store = build_persistent_agent_graph()
    thread_id = str(uuid.uuid4())
    user_id = "user_1"
    print("Persistent memory agent with middleware running.")
    print("Type 'quit' to exit, 'clear cache' or 'clear memory' commands available.")
    while True:
        q = input("\nYour question: ").strip()
        if not q:
            continue
        if q.lower() == "quit":
            break
        if q.lower() == "clear cache":
            clear_cache()
            continue
        if q.lower() == "clear memory":
            store.clear_user_memory(user_id)
            continue
        # set up initial state
        initial_state = {
            "messages": [{"role": "user", "content": q}],
            "profile_memory": None,
            "conversation_memory": None,
            "observation": None,
            "final_answer": None,
            "profile_update": None,
            "conversation_update": None,
            "user_id": user_id,
            "cache_hit": False
        }
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(initial_state, config)
        print("\nAI:", result.get("final_answer"))
        # show memory
        print("\n[Profile memory]:")
        print(get_memory(store, PROFILE_NS, user_id) or "None")
        print("\n[Conversation memory]:")
        print(get_memory(store, CONVO_NS, user_id) or "None")

if __name__ == "__main__":
    run_agent_cli()
