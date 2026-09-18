"""
streamlit_app.py
Browser-based UI for Travel Buddy, wrapping the same AutoGen + Gemini
multi-agent crew used by main.py.
 
Run:
    streamlit run streamlit_app.py
"""
 
import logging
import warnings
 
import streamlit as st
 
# --- Quiet down noisy third-party logs (they'd otherwise clutter the
# terminal Streamlit is running in) ---
logging.getLogger("autogen.oai.client").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
 
from agents import (
    city_expert,
    group_chat,
    group_chat_manager,
    local_agent,
    travel_agent_manager,
    user_proxy,
)
from utils import build_trip_request, extract_final_itinerary, save_itinerary_file
 
ALL_AGENTS = [user_proxy, city_expert, local_agent, travel_agent_manager, group_chat_manager]
 
 
def reset_agents() -> None:
    """AutoGen agents keep their conversation history between calls.
    Streamlit keeps this Python process (and the agent objects) alive
    across form submissions, so without an explicit reset, a second trip
    request in the same browser session would still carry the first
    trip's conversation in context. Clear everyone before each new run."""
    for agent in ALL_AGENTS:
        agent.reset()
    group_chat.messages.clear()
 
 
st.set_page_config(page_title="Travel Buddy", page_icon="🧳", layout="centered")
 
st.title("🧳 Travel Buddy")
st.caption(
    "A multi-agent AI trip planner — three AutoGen agents (city research, "
    "local guide, itinerary manager) powered by Google Gemini."
)
 
with st.form("trip_form"):
    destination = st.text_input("Destination", placeholder="e.g. Goa, London, Tokyo")
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Number of days", min_value=1, max_value=30, value=7)
    with col2:
        budget = st.text_input("Total budget", placeholder="e.g. $1200 or \u20b940000")
    submitted = st.form_submit_button("Plan my trip \u2708\ufe0f")
 
if submitted:
    if not destination.strip() or not budget.strip():
        st.warning("Please fill in both destination and budget.")
    else:
        reset_agents()
        request_message = build_trip_request(destination, days, budget)
 
        with st.spinner("Agents are researching weather, costs, and local spots... this can take 30-60s"):
            chat_result = user_proxy.initiate_chat(group_chat_manager, message=request_message)
 
        itinerary = extract_final_itinerary(chat_result.chat_history)
 
        if itinerary:
            st.success("Your itinerary is ready!")
            st.markdown(itinerary)
 
            saved_path = save_itinerary_file(destination, itinerary)
            st.download_button(
                "\u2b07\ufe0f Download as Markdown",
                data=itinerary,
                file_name=saved_path.name,
                mime="text/markdown",
            )
        else:
            st.error(
                "Something went wrong — the agents didn't produce a final "
                "itinerary. Check the terminal running `streamlit run` for "
                "the full agent conversation and any errors."
            )
 