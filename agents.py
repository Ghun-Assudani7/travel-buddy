"""
agents.py
 
Defines the multi-agent crew for Travel Buddy:
- city_expert      : researches weather, events, and travel costs for a city
- local_agent      : compiles local attractions, food, and customs
- travel_agent_manager : synthesizes both into a final day-by-day itinerary
- user_proxy       : represents the human, kicks off the conversation
 
All four agents share one Gemini-backed llm_config so the whole crew
runs on a single API key with no OpenAI dependency.
"""
 
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
 
from config import GEMINI_API_KEY
from gemini_client import GeminiModelClient
 
llm_config = {
    "config_list": [
        {
            "model": "gemini-3.6-flash",
            "model_client_cls": "GeminiModelClient",
            "api_key": GEMINI_API_KEY,
        }
    ],
    "cache_seed": None,  # disable caching so each run hits the live model
}
 
 
def _with_gemini(agent: AssistantAgent) -> AssistantAgent:
    """Attach the Gemini backend to an agent and return it (for chaining)."""
    agent.register_model_client(model_client_cls=GeminiModelClient)
    return agent
 
 
travel_agent_manager = _with_gemini(AssistantAgent(
    name="travel_agent_manager",
    llm_config=llm_config,
    system_message=(
        "You are the travel agent manager. Combine input from city_expert "
        "and local_agent into ONE complete itinerary matching the exact "
        "trip length and budget the user requested: a day-by-day "
        "plan, packing suggestions, and a rough budget breakdown. "
        "If city_expert flagged the budget as unrealistic, lead with that "
        "warning and either suggest a scaled-down version of the trip "
        "(fewer days, cheaper dates, budget accommodation) or state the "
        "minimum budget needed — do not silently invent a plan that "
        "wouldn't actually work. "
        "Be specific — name real neighborhoods, attractions, and food spots "
        "based on what the other agents reported. "
        "End your final message with the word TERMINATE."
    ),
))
 
city_expert = _with_gemini(AssistantAgent(
    name="city_expert",
    llm_config=llm_config,
    system_message=(
        "You are a city research expert. For the requested destination, "
        "report: typical weather for the travel dates, notable seasonal "
        "events/festivals, and a realistic cost estimate (flights + stay). "
        "Keep it factual and concise. If the user's stated budget is "
        "clearly too low for the destination and trip length, say so "
        "explicitly and state the minimum realistic budget instead of "
        "forcing a plan to fit."
    ),
))
 
local_agent = _with_gemini(AssistantAgent(
    name="local_agent",
    llm_config=llm_config,
    system_message=(
        "You are a local guide. Provide key attractions, food spots, "
        "local customs/etiquette, and day-by-day activity ideas for the "
        "requested destination. Prioritize specific, lesser-known spots "
        "over generic tourist lists."
    ),
))
 
user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config=False,
    is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
)
 
group_chat = GroupChat(
    # Order matters here: round_robin cycles through this list in order,
    # so put it user -> researchers -> synthesizer.
    agents=[user_proxy, city_expert, local_agent, travel_agent_manager],
    messages=[],
    max_round=8,
    # "auto" (the default) uses an LLM call to pick the next speaker, via
    # an internal helper agent AutoGen creates on the fly. That internal
    # agent can't have our custom GeminiModelClient registered on it,
    # which raises "model client(s) are not activated". round_robin
    # needs no extra LLM call, so it sidesteps that entirely — and for
    # this fixed 4-agent pipeline, a fixed order is exactly what we want.
    speaker_selection_method="round_robin",
)
 
group_chat_manager = _with_gemini(GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in (msg.get("content") or ""),
))