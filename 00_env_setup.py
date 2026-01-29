import sys 
import platform 
from huggingface_hub import HfApi

print("Python version:", sys.version)
print("Platform:", platform.platform())
import os
from dotenv import load_dotenv

load_dotenv()

env_vars = [
    "HF_TOKEN",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
    "REDIS_HOST",
    "REDIS_PORT",
]

for var in env_vars:
    value = os.getenv(var)
    if value is None:
        raise RuntimeError(f" Missing environment variable: {var}")
    print(f" {var} loaded")
# LangSmith Tracing Configuration

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

print(" LangSmith tracing configured")

# Hugging Face Inference API Connectivity Test
#load available models 


api = HfApi()

models = api.list_models(author="TheBloke", filter="text-generation")
for model in models:
    print(f"- {model.modelId}")
    
models_iterator = api.list_models(limit=20, filter = "text-generation")
# Convert the iterator to a list to iterate over them and print
models_list = list(models_iterator)

print(f"Found {len(models_list)} models (first 20 results):")
for model in models_list:
    print(f"- {model.modelId}")
from huggingface_hub import InferenceClient
hf_client = InferenceClient(token=os.getenv("HF_TOKEN"))

response = hf_client.chat_completion(
    model="Qwen/Qwen2.5-72B-Instruct",
    messages=[{"role": "user", "content": "Explain the Einstein's theory of Energy '"}],

)

print("HF API response:")
print(response["choices"][0]["message"]["content"])
print(" === Full response ===")
for key, value in response.items():
    print(f"{key}: {value}")
# LangSmith Trace Validation
from langchain_core.language_models.llms import LLM
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client as LangSmithClient
from langchain_core.callbacks import CallbackManager

client = LangSmithClient()
try :
    projects = client.list_projects()
    print(" LangSmith projects:")
    for project in projects:
        print(f"- {project.name} (ID: {project.id})")
except Exception as e:
    print(" Error connecting to LangSmith:", str(e))
    print(" Please check your LANGCHAIN_API_KEY and network connection.")
with tracing_v2_enabled():
    llm = LLM.from_model_id(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        client=hf_client,
        temperature=0.7,
        max_tokens=256,
    )

    prompt = "Explain the theory of relativity in simple terms."
    result = llm.invoke(prompt)

    print("LangChain LLM response:")
    print(result) 
    
#LangChain Built-In Memory Check (InMemoryStore)
from langgraph.store.memory import InMemoryStore


memory_store = InMemoryStore()

namespace = ("test_user", "test_context")
memory_store.put(namespace, "sample_key", {"data": "sample_value"})
found = memory_store.get(namespace, "sample_key")
print("Retrieved from InMemoryStore:", found)
#LangChain In-Memory Chat Memory (short-term memory)
try:
    from langgraph.checkpoint.memory import InMemorySaver
    saver = InMemorySaver()
    print(" InMemorySaver available for short-term memory (threaded).")
except Exception as e:
    print("Short-term memory InMemorySaver import failed:", repr(e))
    print("Install langgraph-checkpoint packages if needed.")

try:
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()
    ns = ("demo_user", "session")
    store.put(ns, "key1", {"text": "long term memory test"})
    val = store.get(ns, "key1")
    print(" InMemoryStore long-term memory works:", val)
except Exception as e:
    print("Long-term memory InMemoryStore import failed:", repr(e))
