"""
utils.py
Shared helpers used by both main.py (CLI) and streamlit_app.py (web UI):
building the trip request message, pulling the final itinerary out of
the group chat history, and saving it to a Markdown file.
"""
 
import re
from datetime import datetime
from pathlib import Path
 
OUTPUT_DIR = Path("outputs")
 
 
def build_trip_request(destination: str, days, budget: str) -> str:
    return (
        f"I want to plan a {days}-day trip to {destination}. "
        f"My budget is {budget}. Include a day-by-day itinerary, "
        f"weather info, local customs, places to visit, food "
        f"recommendations, and what to pack."
    )
 
 
def extract_final_itinerary(chat_history: list[dict]) -> str | None:
    """Pull travel_agent_manager's last message (the synthesized
    itinerary) out of the group chat history, with TERMINATE stripped."""
    for msg in reversed(chat_history):
        if msg.get("name") == "travel_agent_manager" and msg.get("content"):
            return msg["content"].replace("TERMINATE", "").strip()
    return None
 
 
def save_itinerary_file(destination: str, itinerary_text: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", destination.lower()).strip("-") or "trip"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = OUTPUT_DIR / f"{slug}-itinerary-{timestamp}.md"
    out_path.write_text(itinerary_text, encoding="utf-8")
    return out_path
 