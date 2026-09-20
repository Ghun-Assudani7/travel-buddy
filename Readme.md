Travel Buddy — Multi-Agent AI Trip Planner

A multi-agent system built with Microsoft AutoGen and Google Gemini that plans a full 7-day travel itinerary through a coordinated crew of AI agents, each with a distinct role.

How it works
User request
     │
     ▼
 user_proxy  ──►  GroupChatManager  ──►  routes between:
                                            ├── city_expert        (weather, events, cost)
                                            ├── local_agent        (attractions, food, customs)
                                            └── travel_agent_manager (synthesizes final itinerary)

GroupChatManager lets the agents converse and hand off to each other automatically (no manual orchestration code) until travel_agent_manager produces a final itinerary and signals TERMINATE.

Key engineering points
Custom LLM integration: AutoGen's config_list is built around OpenAI-shaped APIs. gemini_client.py implements AutoGen's ModelClient protocol (create, message_retrieval, cost, get_usage) to plug Google Gemini into the framework instead.
Secrets management: API key is loaded from a local .env file via python-dotenv — never hardcoded or committed.
Current, non-deprecated SDKs: uses google-genai (the actively maintained SDK), not the sunset google-generativeai package.
Tech stack

Python · AutoGen (multi-agent orchestration) · Google Gemini API · Streamlit (web UI) · python-dotenv

Setup
bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

copy .env.example .env         # Windows
# cp .env.example .env         # macOS/Linux
# then edit .env and add your GEMINI_API_KEY
# get a free key: https://aistudio.google.com/apikey

Run in the terminal:

bash
python main.py

Or run the browser UI:

bash
streamlit run streamlit_app.py

Both save the final itinerary as a Markdown file in outputs/.

Example output

Running python main.py (or the Streamlit UI) asks for a destination, trip length, and budget, then plans the whole trip end-to-end: weather and cost research, local attraction/food recommendations, and a final day-by-day itinerary with a packing list and budget breakdown — all generated autonomously by the agent crew. If the budget is unrealistic for the destination, the agents flag that honestly instead of forcing a plan that wouldn't actually work.

![image alt](https://raw.githubusercontent.com/Ghun-Assudani7/travel-buddy/refs/heads/main/Assets/Asset.png)


![Streamlit UI](assets/Asset.png)
![Streamlit UI](assets/Asset1.png)

Possible extensions
Add a real search_internet tool call (e.g. Tavily/SerpAPI) so city_expert and local_agent use live data instead of the model's own knowledge.
Add cost/token tracking in GeminiModelClient.cost() for budget monitoring across runs.
Deploy the Streamlit app publicly (with rate limiting, since it would run on a shared API key).
