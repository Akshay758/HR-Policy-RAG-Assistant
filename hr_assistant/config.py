"""All settings for the app live here, in one place."""

import os 
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")


## DIFINE THE PATH - DATA / VECTOR STORE 

FILE_PATH = os.path.join("data", "hr_policy.txt")


VECTOR_STORE_PATH = os.path.join("data", "faiss_index")

LLLM_MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_MODEL_NAME = "jina-embeddings-v5-text-small"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K_RESULT = 2

### SYSTEM INTRUCTIONS 

SYSTEM_PROMPT = (
    "You are a friendly HR Assistant.always use the search_hr_policy tool to look up"
    "facts before answering.If the answer isn't in the search result, say you don't know "
    "insted of guessing."
)


def check_api_keys() -> None:
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY")
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY")