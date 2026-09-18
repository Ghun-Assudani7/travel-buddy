"""
config.py
Loads secrets (API keys) from a local .env file so they never get
hardcoded or committed to git. Copy `.env.example` to `.env` and fill
in your real key before running the project.
"""
 
import os
from dotenv import load_dotenv
 
load_dotenv()
 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
 
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Create a `.env` file in the project "
        "root (see `.env.example`) and set GEMINI_API_KEY=your_key_here."
    )