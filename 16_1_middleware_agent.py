# persistent_agent_with_middleware.py
"""
Persistent-memory LangGraph agent with:
- semantic cache (shelve + diskcache)
- persistent memory (shelve)
- pluggable middleware (before_model / after_model)
- human-in-the-loop via langgraph interrupts & Command resume flow
- streaming support via graph.astream
- model fallback, call limits, summarization middleware examples
- safe LLM wrapper with metadata and LangSmith trace integration

Usage:
    python persistent_agent_with_middleware.py     # interactive REPL mode (console)
    or import build_graph, run_agent, astream_run and use them programmatically.

Dependencies (pip install if missing):
    langgraph, langsmith, huggingface-hub, ddgs, sentence-transformers,
    diskcache, numpy, shelve
"""

import os
import uuid
import json
import re
import time
import hashlib
import shelve
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from huggingface_hub import InferenceClient
from langsmith.run_helpers import trace
from langsmith import Client as LangSmithClient
from ddgs import DDGS

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command  # interrupt / Command for LangGraph node-level interrupts
from langchain_core.messages import HumanMessage  # used for initial_state message object

# Optional (semantic caching embeddings)
from sentence_transformers import SentenceTransformer

# diskcache for fast exact cache
try:
    from diskcache import Cache as DiskCache
except Exception:
    DiskCache = None

# -----------------------------
# Configuration and environment
# -----------------------------
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "local-project")
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN must be set in environment")

hf_client = InferenceClient(token=HF_TOKEN)
langsmith_client = LangSmithClient() if os.getenv("LANGCHAIN_API_KEY") else None

# Paths
SHELVE_CACHE_PATH = os.getenv("SHELVE_CACHE_PATH", "./query_cache.db")
MEMORY_DB_PATH = os.getenv("MEMORY_DB_PATH", "./persistent_memory.db")
DISKCACHE_DIR = os.getenv("DISKCACHE_DIR", "./cache_dir")

# instantiate caches
_diskcache = DiskCache(DISKCACHE_DIR) if DiskCache is not None else None
_embedding_model = None
try:
    _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    # If embedding unavailable, semantic cache will only do exact matches
    _embedding_model = None

# small helper ------------------------------------------------
def now_ts() -> float:
    return time.time()

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# -----------------------------
# Persistent memory store (shelve wrapper)
# -----------------------------
class PersistentMemoryStore:
    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        # ensure file created
        try:
            with shelve.open(self.db_path) as _:
                pass
        except Exception as e:
            print("Warning: could not initialize persistent memory:", e)

    def _key(self, namespace: Tuple[str, ...], user_id: str) -> str:
        return f"{'|'.join(namespace)}|{user_id}"

    def get(self, namespace: Tuple[str, ...], user_id: str) -> str:
        try:
            with shelve.open(self.db_path) as db:
                return db.get(self._key(namespace, user_id), "")
        except Exception as e:
            print("Memory read error:", e)
            return ""

    def put(self, namespace: Tuple[str, ...], user_id: str, value: str):
        try:
            with shelve.open(self.db_path, writeback=True) as db:
                db[self._key(namespace, user_id)] = value
        except Exception as e:
            print("Memory write error:", e)

    def delete(self, namespace: Tuple[str, ...], user_id: str):
        try:
            with shelve.open(self.db_path, writeback=True) as db:
                key = self._key(namespace, user_id)
                if key in db:
                    del db[key]
        except Exception as e:
            print("Memory delete error:", e)

    def list_users(self) -> List[str]:
        users = set()
        try:
            with shelve.open(self.db_path) as db:
                for k in db.keys():
                    if "|" in k:
                        users.add(k.split("|")[-1])
            return list(users)
        except Exception as e:
            print("Memory list error:", e)
            return []

    def clear_user(self, user_id: str):
        try:
            with shelve.open(self.db_path, writeback=True) as db:
                to_del = [k for k in db.keys() if k.endswith(f"|{user_id}")]
                for k in to_del:
                    del db[k]
        except Exception as e:
            print("Memory clear error:", e)

# -----------------------------
# semantic caching (exact + semantic)
# -----------------------------
def semantic_cache_lookup(query: str, user_id: str, similarity_threshold: float = 0.82) -> Optional[Dict[str, Any]]:
    key = md5_hex(f"{user_id}|{query.strip().lower()}")
    # 1) exact diskcache
    if _diskcache:
        try:
            if key in _diskcache:
                return _diskcache.get(key)
        except Exception:
            pass
    # 2) shelve semantic scan if embeddings available
    if _embedding_model:
        q_emb = _embedding_model.encode(query)
        try:
            with shelve.open(SHELVE_CACHE_PATH) as db:
                for k in db.keys():
                    cached = db[k]
                    if cached.get("user_id") != user_id:
                        continue
                    emb = np.array(cached["embedding"])
                    sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
                    if sim >= similarity_threshold:
                        return cached
        except Exception:
            pass
    return None

def semantic_cache_write(query: str, user_id: str, final_answer: str):
    key = md5_hex(f"{user_id}|{query.strip().lower()}")
    data = {
        "query": query,
        "final_answer": final_answer,
        "user_id": user_id,
        "timestamp": now_ts(),
    }
    if _embedding_model:
        try:
            data["embedding"] = _embedding_model.encode(query).tolist()
        except Exception:
            data["embedding"] = []
    try:
        with shelve.open(SHELVE_CACHE_PATH, writeback=True) as db:
            db[key] = data
    except Exception:
        pass
    if _diskcache:
        try:
            _diskcache.set(key, data, expire=3600)
        except Exception:
            pass

# -----------------------------
# Middleware system (lightweight)
# -----------------------------
# Design: middleware classes implement before_model(self, state, prompt) -> (prompt, meta)
# and after_model(self, state, prompt, response) -> response (possibly modified).
# before_model may also return a dict with {"interrupt": {...}} to request a langgraph interrupt.
@dataclass
class MiddlewareResult:
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)
    interrupt_payload: Optional[Any] = None

class BaseMiddleware:
    """
    Base middleware with optional hooks. Implement the hook methods you need.
    Hooks are executed in registration order for before_model and reverse order for after_model.
    """
    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        return MiddlewareResult(prompt=prompt)

    def after_model(self, state: Dict[str, Any], prompt: str, response: str) -> str:
        return response

# -----------------------------
# Example middleware implementations
# -----------------------------
class SummarizationMiddleware(BaseMiddleware):
    """
    If conversation length (messages) exceeds a threshold, summarize using the LLM
    and replace messages contents with the summary to keep token usage low.
    This is executed BEFORE model calls inside a node when run via middleware_manager.run_before_model.
    """
    def __init__(self, token_threshold: int = 3000, summary_model: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.token_threshold = token_threshold
        self.model = summary_model

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        # simple heuristic: sum of string lengths as approximate tokens
        messages = state.get("messages") or []
        total_chars = sum(len(str(m)) for m in messages)
        if total_chars < self.token_threshold:
            return MiddlewareResult(prompt=prompt)
        # request a summary from LLM (synchronous call)
        summary_prompt = (
            "You are a conversation summarizer. "
            "Summarize the conversation below into a short bullet list of facts that should be kept for future context.\n\n"
            "Conversation:\n" + "\n".join([str(m) for m in messages]) + "\n\nSummary:"
        )
        # call LLM directly (no middleware nesting)
        try:
            with trace(name="middleware_summarize", project_name=PROJECT_NAME):
                resp = hf_client.chat_completion(model=self.model, messages=[{"role":"user", "content":summary_prompt}], max_tokens=256)
                summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = "Summary failed."
        # replace prompt with augmenting the prompt with the summary as context
        new_prompt = f"Context Summary:\n{summary}\n\n{prompt}"
        return MiddlewareResult(prompt=new_prompt, meta={"summary": summary})

class HumanInTheLoopMiddleware(BaseMiddleware):
    """
    If a tool call or model call matches a policy condition, request human approval
    by returning an interrupt payload. The node calling middleware_manager.run_before_model
    will call langgraph.types.interrupt(...) and pause.
    """
    def __init__(self, approval_keywords: Optional[List[str]] = None):
        self.approval_keywords = approval_keywords or ["delete account", "transfer", "payment", "execute script"]

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        # If prompt asks to perform a high-risk action, request approval
        text = prompt.lower()
        for kw in self.approval_keywords:
            if kw in text:
                payload = {
                    "question": "Human approval required",
                    "details": f"The prompt seems to request: {kw}. Prompt excerpt: {prompt[:400]}",
                }
                return MiddlewareResult(prompt=prompt, interrupt_payload=payload)
        return MiddlewareResult(prompt=prompt)

class ModelCallLimitMiddleware(BaseMiddleware):
    """
    Track model call counts per thread and per run; enforce limits.
    Uses an in-memory store here; can be persisted via a checkpointer in production.
    """
    def __init__(self, thread_limit: Optional[int] = None, run_limit: Optional[int] = None):
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.thread_counters: Dict[str, int] = {}
        self.run_counters: Dict[str, int] = {}

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        thread_id = state.get("_thread_id")
        run_id = state.get("_run_id")
        if thread_id:
            self.thread_counters.setdefault(thread_id, 0)
            if self.thread_limit is not None and self.thread_counters[thread_id] >= self.thread_limit:
                raise RuntimeError(f"Thread model call limit reached ({self.thread_limit})")
        if run_id:
            self.run_counters.setdefault(run_id, 0)
            if self.run_limit is not None and self.run_counters[run_id] >= self.run_limit:
                raise RuntimeError(f"Run model call limit reached ({self.run_limit})")
        # record increment after actual call in middleware manager (so no change to prompt)
        return MiddlewareResult(prompt=prompt)

    def record_call(self, state: Dict[str, Any]):
        thread_id = state.get("_thread_id")
        run_id = state.get("_run_id")
        if thread_id:
            self.thread_counters[thread_id] = self.thread_counters.get(thread_id, 0) + 1
        if run_id:
            self.run_counters[run_id] = self.run_counters.get(run_id, 0) + 1

class ToolCallLimitMiddleware(BaseMiddleware):
    """
    Track tool calls per thread/run. Tools are represented in state['tool_calls'].
    """
    def __init__(self, tool_name: Optional[str] = None, thread_limit: Optional[int] = None, run_limit: Optional[int] = None):
        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_counters_thread: Dict[str,int] = {}
        self.tool_counters_run: Dict[str,int] = {}

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        # this middleware typically enforces counts prior to tool usage; for simplicity,
        # we check counts recorded in state and raise if over limit
        thread_id = state.get("_thread_id")
        run_id = state.get("_run_id")
        key = f"{thread_id}:{self.tool_name}" if self.tool_name else f"{thread_id}:__all__"
        if self.thread_limit is not None and self.tool_counters_thread.get(key,0) >= self.thread_limit:
            raise RuntimeError("Tool call thread limit reached")
        if self.run_limit is not None and self.tool_counters_run.get(f"{run_id}:{self.tool_name}",0) >= self.run_limit:
            raise RuntimeError("Tool call run limit reached")
        return MiddlewareResult(prompt=prompt)

    def record_tool_call(self, state: Dict[str, Any], tool_name: str):
        thread_id = state.get("_thread_id")
        run_id = state.get("_run_id")
        key = f"{thread_id}:{tool_name}"
        self.tool_counters_thread[key] = self.tool_counters_thread.get(key,0) + 1
        key2 = f"{run_id}:{tool_name}"
        self.tool_counters_run[key2] = self.tool_counters_run.get(key2,0) + 1

class ModelFallbackMiddleware(BaseMiddleware):
    """
    If the primary LLM call fails (exception) or returns an explicit failure signal,
    the node can call the fallback models using this middleware logic.
    This class provides an utility method; the actual fallback loop is used in call_llm_with_middlewares.
    """
    def __init__(self, fallback_models: Optional[List[str]] = None):
        self.fallback_models = fallback_models or []

# Minimal PII detector example
import re
class PIIMiddleware(BaseMiddleware):
    """
    Simple PII detector that redacts email addresses and SSN-like patterns before the model sees them.
    Returns modified prompt with redactions and a meta flag.
    """
    EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
    SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        redacted = self.EMAIL_RE.sub("[REDACTED_EMAIL]", prompt)
        redacted = self.SSN_RE.sub("[REDACTED_SSN]", redacted)
        meta = {"pii_redacted": True} if redacted != prompt else {}
        return MiddlewareResult(prompt=redacted, meta=meta)

class ToolRetryMiddleware(BaseMiddleware):
    """
    Wrap tool invocations with retry logic; used by nodes that execute external tools.
    For here, we just supply a helper function; integration should call middleware.retry_tool_call.
    """
    def __init__(self, retries: int = 3, backoff: float = 1.0):
        self.retries = retries
        self.backoff = backoff

    def retry_call(self, func: Callable, *args, **kwargs):
        last_exc = None
        for attempt in range(self.retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                time.sleep(self.backoff * (2 ** attempt))
        raise last_exc

class LLMToolEmulator(BaseMiddleware):
    """
    Emulates tool outputs by asking the LLM for expected tool results. Useful in tests.
    If emulation is enabled, before_model returns an 'emulate_tool' meta that downstream code checks.
    """
    def __init__(self, enabled: bool = False, emulator_model: str = "Qwen/Qwen2.5-72B-Instruct"):
        self.enabled = enabled
        self.emulator_model = emulator_model

    def before_model(self, state: Dict[str, Any], prompt: str) -> MiddlewareResult:
        if not self.enabled:
            return MiddlewareResult(prompt=prompt)
        emulate_prompt = f"Emulate tool behavior for: {prompt}\nReturn a short simulation output."
        try:
            with trace(name="tool_emulator", project_name=PROJECT_NAME):
                resp = hf_client.chat_completion(model=self.emulator_model, messages=[{"role":"user","content":emulate_prompt}], max_tokens=128)
                sim = resp.choices[0].message.content.strip()
        except Exception:
            sim = "EMULATION_FAILED"
        return MiddlewareResult(prompt=prompt, meta={"emulated_tool_output": sim})

# -----------------------------
# Middleware manager
# -----------------------------
class MiddlewareManager:
    def __init__(self, middleware: List[BaseMiddleware]):
        self._list = middleware

    def run_before_model(self, state: Dict[str, Any], prompt: str) -> Tuple[str, Dict[str,Any], Optional[Any]]:
        """
        Execute before_model for each middleware in order.
        Returns (prompt, accumulated_meta, interrupt_payload)
        If a middleware returns interrupt_payload, return it immediately (node will call interrupt())
        """
        meta: Dict[str, Any] = {}
        current_prompt = prompt
        for mw in self._list:
            try:
                res = mw.before_model(state, current_prompt)
                if not isinstance(res, MiddlewareResult):
                    res = MiddlewareResult(prompt=current_prompt)
                # update prompt & meta
                current_prompt = res.prompt or current_prompt
                meta.update(res.meta or {})
                if res.interrupt_payload is not None:
                    return current_prompt, meta, res.interrupt_payload
            except Exception as e:
                # middleware bug should not kill whole agent — log and continue
                print(f"[Middleware before_model error] {type(mw).__name__}: {e}")
                traceback.print_exc()
        return current_prompt, meta, None

    def run_after_model(self, state: Dict[str, Any], prompt: str, response: str) -> str:
        """
        Execute after_model hooks in reverse order.
        """
        cur = response
        for mw in reversed(self._list):
            try:
                cur = mw.after_model(state, prompt, cur)
            except Exception as e:
                print(f"[Middleware after_model error] {type(mw).__name__}: {e}")
                traceback.print_exc()
        return cur

# -----------------------------
# LLM wrapper that integrates middleware manager and model fallback and LangSmith trace
# -----------------------------
def call_llm_with_middlewares(state: Dict[str, Any], prompt: str, middleware_manager: MiddlewareManager, *,
                              model: str = "Qwen/Qwen2.5-72B-Instruct", max_tokens: int = 512,
                              fallback_models: Optional[List[str]] = None) -> Tuple[str, Optional[str]]:
    """
    Execute middleware before_model hooks, possibly trigger interrupt via langgraph.types.interrupt,
    call the LLM, then run after_model hooks. Returns (response_text, run_id)
    This function SHOULD be called from inside a LangGraph node function so interrupts are valid.
    """
    # run before_model hooks
    prompt_after, meta, interrupt_payload = middleware_manager.run_before_model(state, prompt)
    if interrupt_payload is not None:
        # call langgraph interrupt -> pause node & return when resumed by Command
        # interrupt(...) returns the resume payload when the run is resumed
        # The docs show interrupt(...) used directly inside a node, so we call it here.
        resumed_value = interrupt(interrupt_payload)  # this pauses the node until resume
        # The resumed_value becomes available and you may want to use it to update the prompt
        # If resumed_value is dict or str, integrate into prompt
        # If None, proceed unchanged
        if resumed_value:
            prompt_after = f"{prompt_after}\n\nHuman approval/resume: {resumed_value}"
    # wrap actual model call with trace
    attempt_models = [model] + (fallback_models or [])
    last_exc = None
    run_id = None
    for m in attempt_models:
        try:
            with trace(name="llm_call", project_name=PROJECT_NAME, metadata={"model": m}):
                resp = hf_client.chat_completion(model=m, messages=[{"role":"user","content":prompt_after}], max_tokens=max_tokens)
                run_id = getattr(resp, "id", None)
                # extract text safely (supports HF chat response style)
                text = ""
                try:
                    text = resp.choices[0].message.content or ""
                except Exception:
                    # fallback if structure differs
                    if isinstance(resp, dict):
                        # some wrappers return dicts
                        text = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        text = str(resp)
                # run after_model hooks
                text = middleware_manager.run_after_model(state, prompt_after, text)
                # record counters on certain middleware (if present)
                for mw in middleware_manager._list:
                    if isinstance(mw, ModelCallLimitMiddleware):
                        try:
                            mw.record_call(state)
                        except Exception:
                            pass
                return text, run_id
        except Exception as e:
            last_exc = e
            # continue to next fallback model
            continue
    # If all fallback attempts failed, raise the last exception
    if last_exc:
        raise last_exc
    return "", run_id

# -----------------------------
# Agent graph nodes (LangGraph)
# -----------------------------
PROFILE_NS = ("user_profile",)
CONVO_NS = ("user_conversation",)

# Build a default middleware set (customize/replace instantly)
DEFAULT_MIDDLEWARE = [
    PIIMiddleware(),
    SummarizationMiddleware(token_threshold=3000),
    HumanInTheLoopMiddleware(approval_keywords=["transfer", "delete account", "execute payment"]),
    ModelCallLimitMiddleware(thread_limit=None, run_limit=20),
    ToolCallLimitMiddleware(tool_name=None, thread_limit=50, run_limit=20),
    ToolRetryMiddleware(retries=2, backoff=0.5),
    LLMToolEmulator(enabled=False),
]
middleware_manager = MiddlewareManager(DEFAULT_MIDDLEWARE)

# Node state typing is dynamic dict here; LangGraph will accept TypedDict, but we keep it simple
def memory_load_node(state: Dict[str,Any], store: PersistentMemoryStore):
    # load profile + conversation memory into state
    return {
        "profile_memory": store.get(PROFILE_NS, state["user_id"]) or "",
        "conversation_memory": store.get(CONVO_NS, state["user_id"]) or "",
        "_thread_id": state.get("_thread_id"),
        "_run_id": state.get("_run_id"),
    }

def planner_node(state: Dict[str,Any]) -> Dict[str,Any]:
    # simple planner using LLM
    last = state["messages"][-1]
    user_query = last.content if isinstance(last, HumanMessage) else last.get("content", str(last))
    prompt = f"Break the user's question into up to 2 actionable steps. Question: {user_query}\nReturn a JSON array of steps."
    out, run_id = call_llm_with_middlewares(state, prompt, middleware_manager)
    # extract json
    try:
        plan = json.loads(out)
        if not isinstance(plan, list):
            raise ValueError("Plan not list")
    except Exception:
        plan = [user_query]
    return {"plan": plan[:2], "current_step": 0, "observations": [], "_last_run_id": run_id}

def action_node(state: Dict[str,Any], store: PersistentMemoryStore) -> Dict[str,Any]:
    # run before model middleware — particularly human approval can occur here
    last = state["messages"][-1]
    user_query = last.content if isinstance(last, HumanMessage) else last.get("content", str(last))
    step = (state.get("plan") or [user_query])[state.get("current_step", 0)]
    # create the action-level prompt that will be passed through middleware
    prompt = f"Execute the step: {step}\nUser memory: {state.get('profile_memory','')}\nConversation memory: {state.get('conversation_memory','')}\nIf you cannot answer, return the word INSUFFICIENT."
    # Use middleware_manager inside node so interrupts are valid
    # call model with middleware integration
    observation, run_id = call_llm_with_middlewares(state, prompt, middleware_manager)
    # If LLM returned 'insufficient' -> fallback to web_search
    if "insufficient" in (observation or "").lower():
        try:
            with trace(name="web_search", project_name=PROJECT_NAME):
                parts = []
                with DDGS() as ddgs:
                    for r in ddgs.text(step, max_results=2):
                        parts.append(f"{r.get('title','')}\n{r.get('body','')}\nSource: {r.get('href','')}")
                observation = "\n\n".join(parts) if parts else "No results found."
        except Exception as e:
            observation = f"Search failed: {e}"
    return {"observations": state.get("observations",[]) + [observation], "_last_run_id": run_id}

def thought_node(state: Dict[str,Any]) -> Dict[str,Any]:
    # short reflection
    step = (state.get("plan") or [])[state.get("current_step",0)]
    observation = (state.get("observations") or [""])[-1]
    prompt = f"Briefly (1-2 sentences) reflect on what was learned for step: {step}\nObservation: {observation[:400]}"
    out, run_id = call_llm_with_middlewares(state, prompt, middleware_manager)
    return {"thought": out.strip()}

def update_step_node(state: Dict[str,Any]) -> Dict[str,Any]:
    return {"current_step": state.get("current_step",0) + 1}

def router(state: Dict[str,Any]) -> str:
    if state.get("current_step",0) < len(state.get("plan",[])):
        return "action"
    return "final_answer"

def final_answer_node(state: Dict[str,Any], store: PersistentMemoryStore) -> Dict[str,Any]:
    last = state["messages"][-1]
    user_query = last.content if isinstance(last, HumanMessage) else last.get("content", str(last))
    all_info = "\n\n".join(state.get("observations",[]))
    prompt = f"Synthesize a clear final answer to: {user_query}\nUse only the information below:\n{all_info}\nIf insufficient, say so."
    answer, run_id = call_llm_with_middlewares(state, prompt, middleware_manager)
    # construct memory update prompts
    profile_prompt = f"Extract ONLY explicit personal facts about the user from: {user_query}\nReturn NONE if nothing to store."
    profile_update, _ = call_llm_with_middlewares(state, profile_prompt, middleware_manager)
    convo_prompt = f"Summarize in one sentence: User asked: {user_query} Assistant answered: {answer}"
    convo_update, _ = call_llm_with_middlewares(state, convo_prompt, middleware_manager)
    # write to persistent memory (append)
    if profile_update and profile_update.strip().lower() not in ["none", "no", ""]:
        existing = store.get(PROFILE_NS, state["user_id"]) or ""
        store.put(PROFILE_NS, state["user_id"], (existing + "\n" + profile_update).strip() if existing else profile_update.strip())
    if convo_update and convo_update.strip().lower() not in ["none", "no", ""]:
        existing2 = store.get(CONVO_NS, state["user_id"]) or ""
        store.put(CONVO_NS, state["user_id"], (existing2 + "\n" + convo_update).strip() if existing2 else convo_update.strip())
    # save to semantic cache for future
    try:
        semantic_cache_write(last.content if isinstance(last, HumanMessage) else last.get("content", ""), state["user_id"], answer)
    except Exception:
        pass
    return {"final_answer": answer, "profile_update": profile_update, "conversation_update": convo_update, "_last_run_id": run_id}

# -----------------------------
# Graph builder and runner
# -----------------------------
def build_graph_with_persistent_memory() -> Tuple[StateGraph, PersistentMemoryStore]:
    store = PersistentMemoryStore(MEMORY_DB_PATH)
    builder = StateGraph(dict)  # use flexible dict state
    builder.add_node("memory_load", lambda s: memory_load_node(s, store))
    builder.add_node("planner", planner_node)
    builder.add_node("action", lambda s: action_node(s, store))
    builder.add_node("thought", thought_node)
    builder.add_node("update_step", update_step_node)
    builder.add_node("final_answer", lambda s: final_answer_node(s, store))
    builder.add_edge(START, "memory_load")
    builder.add_edge("memory_load", "planner")
    builder.add_edge("planner", "action")
    builder.add_edge("action", "thought")
    builder.add_edge("thought", "update_step")
    # conditional router
    builder.add_conditional_edges("update_step", router, {"action": "action", "final_answer": "final_answer"})
    builder.add_edge("final_answer", END)
    graph = builder.compile(checkpointer=MemorySaver(), store=store)
    return graph, store

def run_agent(query: str, graph: StateGraph, store: PersistentMemoryStore, thread_id: str, user_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "plan": None,
        "current_step": 0,
        "observations": [],
        "thought": None,
        "final_answer": None,
        "_thread_id": thread_id,
        "_run_id": str(uuid.uuid4()),
    }
    # Attach run/thread ids to state for middleware that cares
    result = graph.invoke(initial_state, config)
    print("\nFINAL ANSWER:\n", result.get("final_answer"))
    print("\nPROFILE MEMORY:\n", store.get(PROFILE_NS, user_id) or "None")
    print("\nCONVERSATION MEMORY:\n", store.get(CONVO_NS, user_id) or "None")
    return result

# Streaming runner (async) -- demonstrates streaming & interrupt handling as in docs
async def astream_run(initial_input: Dict[str, Any], graph: StateGraph, config: Dict[str, Any]):
    """
    Example of streaming run. The consumer must handle chunked messages and interrupts.
    See docs: graph.astream(...) usage.
    """
    # stream_mode chooses what to stream; messages = model messages, updates = state updates / interrupts
    async for metadata, mode, chunk in graph.astream(initial_input, stream_mode=["messages", "updates"], subgraphs=True, config=config):
        if mode == "messages":
            # chunk is a (message, index) or a chunked message — forward to UI
            msg, idx = chunk
            # Many kinds of message types; show content when available
            try:
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            except Exception:
                content = None
            if content:
                # you should present to user in real time
                print(content, end="", flush=True)
        elif mode == "updates":
            # updates include routing, interrupts, node transitions
            if "__interrupt__" in chunk:
                interrupt_info = chunk["__interrupt__"][0].value
                # Here, present to user and collect response (synchronously or via UI)
                print("\n--- INTERRUPT: human approval required ---")
                print(interrupt_info)
                # For this demo, we accept automatically (or you can accept input())
                user_resp = input("Human response (type 'yes'/'no' or free text): ")
                # resume the graph by invoking again with Command(resume=...)
                # the config must include same thread id
                resumed = graph.invoke(Command(resume=user_resp), config=config)
                return resumed  # we return resumed run result
            else:
                # other updates (node transitions)
                # chunk could contain node state, print keys for debugging
                print("\n[UPDATE]", list(chunk.keys()))
    return None

# -----------------------------
# Example CLI main
# -----------------------------
if __name__ == "__main__":
    graph, store = build_graph_with_persistent_memory()
    thread_id = str(uuid.uuid4())
    user_id = "user_1"
    print("Persistent memory agent with middleware and interrupts.")
    print("Type 'quit' to exit.")
    while True:
        q = input("\nYour question: ").strip()
        if not q:
            continue
        if q.lower() == "quit":
            break
        try:
            run_agent(q, graph, store, thread_id, user_id)
        except Exception as e:
            print("Agent run failed:", e)
            traceback.print_exc()
