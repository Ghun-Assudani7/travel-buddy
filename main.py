"""
main.py
Entry point for Travel Buddy — a multi-agent AI trip planner built on
AutoGen, powered by Google Gemini.
 
Run:
    python main.py
Then answer the destination/days/budget prompts. The final itinerary
is also saved as a Markdown file in the `outputs/` folder.
 
For a browser-based version instead, run: streamlit run streamlit_app.py
"""
 
import logging
import warnings
 
# --- Quiet down noisy third-party logs so the terminal demo stays clean ---
logging.getLogger("autogen.oai.client").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
 
from agents import user_proxy, group_chat_manager
from utils import build_trip_request, extract_final_itinerary, save_itinerary_file
 
 
def get_trip_details() -> tuple[str, str]:
    """Ask the user for trip details. Returns (destination, request_message)."""
    destination = input("Where do you want to go? ").strip() or "London"
    days = input("How many days? ").strip() or "7"
    budget = input("What's your total budget (e.g. $1200)? ").strip() or "$1200"
    return destination, build_trip_request(destination, days, budget)
 
 
if __name__ == "__main__":
    destination, trip_request = get_trip_details()
 
    chat_result = user_proxy.initiate_chat(
        group_chat_manager,
        message=trip_request,
    )
 
    itinerary = extract_final_itinerary(chat_result.chat_history)
    if itinerary:
        saved_path = save_itinerary_file(destination, itinerary)
        print(f"\nItinerary saved to: {saved_path.resolve()}")
    else:
        print("\n(Could not locate a final itinerary message to save.)")
 