import os 
import json
import time
from typing import List, Dict, Any, Optional
from huggingface_hub import InferenceClient

from langsmith import Client
from langsmith.run_helpers import trace 



assert os.getenv("HF_TOKEN"), "HF_TOKEN not set"
assert os.getenv("LANGCHAIN_API_KEY"), "LANGCHAIN_API_KEY not set"
assert os.getenv("LANGCHAIN_PROJECT"), "LANGCHAIN_PROJECT not set"

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))


# ===================== LLM CALL =====================

def call_llm(prompt: str, name: str) -> str:
    with trace(name=name, project_name=PROJECT_NAME):
        resp = hf_client.chat_completion(
            model="Qwen/Qwen2.5-72B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
# ===================== JUDGE PROMPT =====================

EVAL_PROMPT = """
You are an evaluator.

Question:
{question}

Context:
{context}

Answer:
{answer}

Score the answer between 0 and 1 for each metric:

faithfulness (grounded in context)
relevance (answers the question)
quality (clarity and correctness)

Return ONLY valid JSON:
{{
  "faithfulness": number,
  "relevance": number,
  "quality": number
}}
"""
# ===================== EVALUATOR =====================

def evaluate_response(question: str, answer: str, context: str = "") -> dict:
    prompt = EVAL_PROMPT.format(
        question=question,
        context=context if context else "None",
        answer=answer,
    )

    with trace(name="llm_evaluator", project_name=PROJECT_NAME):
        raw = call_llm(prompt, name="judge_llm")
        return json.loads(raw)




# ===================== DEMO =====================

if __name__ == "__main__":
    question = "What is the capital of France?"
    context = "France's capital city is Paris."

    start = time.time()
    answer = call_llm(question, name="main_llm")
    latency = time.time() - start

    scores = evaluate_response(question, answer, context)

    print("\nAnswer:\n", answer)
    print("\nEvaluation Scores:\n", scores)
    print("\nLatency:", round(latency, 3), "seconds")
