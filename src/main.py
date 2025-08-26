from enum import Enum
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
import uvicorn
import logging
from contextlib import asynccontextmanager

from helper import check_time_sensitive
from lru import LRUCache, CacheType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables for models
openai_client = None


class Source(Enum):
    CACHE = "cache"
    LLM = "llm"


class Metadata(BaseModel):
    source: Source


class ChatRequest(BaseModel):
    message: str
    forceRefresh: bool = False


class ChatResponse(BaseModel):
    response: str
    metadata: Metadata


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and clients on startup"""
    global openai_client
    global semantic_cache
    global text_cache

    try:
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        # Fixed: Added proper cache type specification
        semantic_cache = LRUCache(
            item_limit=1000, cache_prefix="semantic_cache", cache_type=CacheType.SEMANTIC
        )
        text_cache = LRUCache(item_limit=1000, cache_prefix="text_cache", cache_type=CacheType.TEXT)

        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.error("OPENAI_API_KEY not found. Please set your OpenAI API key.")
            raise ValueError("OPENAI_API_KEY is required")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise e


app = FastAPI(
    title="Boardy Test API",
    description="A FastAPI chatbot with text embedding and LLM integration",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Chatbot API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "openai_client": openai_client is not None,
        "api_status": "healthy",
    }
    return status


@app.get("/print_metrics")
async def print_metrics():
    """Print metrics endpoint"""
    text_metrics = text_cache.print_metrics()
    semantic_metrics = semantic_cache.print_metrics()
    return {
        "message": "Metrics printed successfully",
        "text_metrics": text_metrics,
        "semantic_metrics": semantic_metrics,
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint"""
    text_metrics = text_cache.get_metrics_dict()
    semantic_metrics = semantic_cache.get_metrics_dict()
    return json.dumps(
        {"text_cache": text_metrics, "semantic_cache": semantic_metrics}, allow_nan=True
    )


@app.post("/api/query")
async def chat(request: ChatRequest) -> ChatResponse:
    """Simplified chat endpoint that returns only the text response"""
    if openai_client is None:
        raise HTTPException(
            status_code=500,
            detail="OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.",
        )

    try:

        if not request.forceRefresh and not check_time_sensitive(request.message):
            ## Check if exact text query is in cache
            if text_cache.has(request.message):
                cached_response = text_cache.get(request.message)
                return {
                    "response": cached_response,
                    "metadata": {"source": "cache"},
                }

            ## Check if similar text query is in semantic cache
            if similar_response := semantic_cache.get_similar(request.message):
                return {
                    "response": similar_response,
                    "metadata": {"source": "cache"},
                }

        # Generate response using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "user", "content": request.message},
            ],
        )

        llm_response = response.choices[0].message.content

        # Store response in both caches
        text_cache.set(request.message, llm_response)
        semantic_cache.set(request.message, llm_response)
        response = {"response": llm_response, "metadata": {"source": "llm"}}

        return response

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 4))

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info", workers=workers)
