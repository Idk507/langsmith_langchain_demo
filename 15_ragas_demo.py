import os
from typing import List

from datasets import Dataset

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings

from langsmith.run_helpers import trace

# =========================================================
# ENV VALIDATION
# =========================================================

assert os.getenv("HF_TOKEN"), "HF_TOKEN missing"
assert os.getenv("LANGCHAIN_API_KEY"), "LANGCHAIN_API_KEY missing"
assert os.getenv("LANGCHAIN_PROJECT"), "LANGCHAIN_PROJECT missing"

PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT")

# =========================================================
# HUGGINGFACE LLM (FOR RAGAS JUDGING)
# =========================================================

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.0,
    max_new_tokens=512,
)

ragas_llm = LangchainLLMWrapper(llm)

# =========================================================
# EMBEDDING MODEL (FOR CONTEXT METRICS)
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# =========================================================
# SAMPLE RAG OUTPUT DATASET
# (This normally comes from your RAG pipeline)
# =========================================================

questions: List[str] = [
    "What is the capital of France?",
    "Who founded OpenAI?",
]

answers: List[str] = [
    "The capital of France is Paris.",
    "OpenAI was founded by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, and others.",
]

contexts: List[List[str]] = [
    ["France's capital city is Paris. It is known for the Eiffel Tower."],
    ["OpenAI was founded in 2015 by Sam Altman, Elon Musk, Greg Brockman, and Ilya Sutskever."],
]

ground_truths: List[str] = [
    "Paris",
    "Sam Altman, Elon Musk, Greg Brockman, and Ilya Sutskever founded OpenAI in 2015.",
]

dataset = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
)

# =========================================================
# RAGAS EVALUATION
# =========================================================

with trace(
    name="ragas_evaluation_run",
    project_name=PROJECT_NAME,
    tags=["ragas", "evaluation"],
):
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

# =========================================================
# OUTPUT RESULTS
# =========================================================

df = result.to_pandas()

print("\n=== RAGAS METRICS ===\n")
print(df)

print("\n=== AVERAGE SCORES ===\n")
print(df.mean(numeric_only=True))
