#!/usr/bin/env python3
"""
Simple script to start the FastAPI server
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Start the FastAPI server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 3000))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    workers = int(os.getenv("WORKERS", 4))
    print(f"🚀 Starting FastAPI Chatbot Server on {host}:{port}")
    print(f"📊 Log level: {log_level}")
    print("🔧 Make sure your .env file contains OPENAI_API_KEY")
    print("📖 Access API docs at: http://localhost:3000/docs")
    print("🏥 Health check at: http://localhost:3000/health")
    print("\n" + "=" * 50 + "\n")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level=log_level, workers=workers)


if __name__ == "__main__":
    main()
