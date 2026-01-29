import os
import time
from typing import Dict, Any

from huggingface_hub import InferenceClient
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import trace

required_env = [
    "HF_TOKEN",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
]

missing = [k for k in required_env if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing environment variables: {missing}")

print(" Required environment variables are present")

hf_client = InferenceClient(
    token=os.getenv("HF_TOKEN")
)

langsmith_client = LangSmithClient()

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

print("Hugging Face client initialized")
print(" LangSmith client initialized")
print("LangSmith Project:", PROJECT_NAME)

# LLM Call Wrapper
from typing import Dict, Any
import time
from langsmith.run_helpers import trace


def call_llm(
    prompt: str,
    *,
    model: str = "zai-org/GLM-4.7-Flash",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    metadata: Dict[str, Any] | None = None,
    tags: list[str] | None = None,
):
    """
    Canonical LLM call wrapper using:
    - Hugging Face Inference chat_completion
    - LangSmith tracing
    - Chat-based models (GLM / OpenAI-style)
    """
    metadata = metadata or {}
    tags = tags or []

    with trace(
        name="hf_llm_call",
        project_name=PROJECT_NAME,
        metadata={
            **metadata,
            "model": model,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        tags=tags,
    ):
        start = time.time()

        response = hf_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )

        latency = round(time.time() - start, 3)

        # Optional: attach latency as metadata
        trace_context = {
            "latency_sec": latency,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            
        }

        return response

def assistant_text(response) -> str :
    if response.choices[0].message.reasoning_content:
        return response.choices[0].message.reasoning_content
    if response.choices[0].message.content:
        return response.choices[0].message.content
prompt = "Say hello and confirm LangSmith tracing works."

response = call_llm(
    prompt,
    metadata={
        "notebook": "01_llm_and_langsmith",
        "purpose": "smoke_test",
    },
    tags=["smoke-test", "llm"],
)

print("Assistant output:\n")
print(assistant_text(response))

print(response.choices[0].message.reasoning_content)

usage = response.usage

print("Token usage:")
print("Prompt tokens:", usage.prompt_tokens)
print("Completion tokens:", usage.completion_tokens)
print("Total tokens:", usage.total_tokens)

questions = [
    "What is an AI agent?",
    "What is ReAct in agentic AI?",
    "What does AoT (Action Observe Thought) mean?",
]

for q in questions:
    resp = call_llm(
        q,
        max_new_tokens=128,
        metadata={
            "question": q,
            "batch": "definitions",
        },
        tags=["batch", "definition"],
    )

    print(f"\nQ: {q}")
    print("A:", assistant_text(resp), "...")

try:
    _ = call_llm(
        None,  # invalid input
        metadata={"test": "error_case"},
        tags=["error-test"],
    )
except Exception as e:
    print("Expected error captured:", type(e).__name__)

fast_response = call_llm(
    "Explain observability in one sentence.",
    max_new_tokens=64,
    temperature=0.2,
    metadata={"mode": "fast"},
    tags=["latency-test"],
)

print(assistant_text(fast_response))
