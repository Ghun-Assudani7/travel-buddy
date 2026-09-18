"""
gemini_client.py
 
A custom AutoGen ModelClient that routes agent calls to Google's Gemini
API via the current `google-genai` SDK.
 
Why this file exists / what it demonstrates:
- AutoGen's built-in `config_list` only understands OpenAI-shaped APIs.
  To use a different provider (Gemini) you implement the `ModelClient`
  protocol and register it on each agent with
  `agent.register_model_client(model_client_cls=GeminiModelClient)`.
- The old `google.generativeai` package is deprecated (sunset by Google);
  this uses its replacement, `google-genai` (`from google import genai`).
"""
 
import time
 
from google import genai
from autogen import ModelClient
 
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2  # doubles each attempt: 2s, 4s, 8s...
 
 
class _Message:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
 
 
class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)
 
 
class _GeminiResponse:
    """Minimal object satisfying AutoGen's ModelClientResponseProtocol."""
 
    def __init__(self, content: str, model: str):
        self.choices = [_Choice(content)]
        self.model = model
 
 
class GeminiModelClient(ModelClient):
    """AutoGen-compatible wrapper around the Gemini API."""
 
    def __init__(self, config, **kwargs):
        self.model_name = config.get("model", "gemini-3.6-flash")
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("GeminiModelClient: no api_key found in llm_config.")
        self._client = genai.Client(api_key=self.api_key)
 
    def create(self, params) -> _GeminiResponse:
        messages = params.get("messages", [])
        # Flatten the chat history into a single prompt. Good enough for
        # AutoGen's turn-based agent conversations; swap for the native
        # multi-turn `contents` format if you need finer control later.
        prompt = "\n\n".join(
            f'{m.get("role", "user").upper()}: {m.get("content", "")}'
            for m in messages
        )
 
        # Long-running requests can hit transient network drops
        # (RemoteProtocolError / connection reset) — retry a few times
        # with backoff before giving up, instead of crashing the whole
        # agent conversation on a one-off blip.
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = (response.text or "").strip()
                return _GeminiResponse(text, self.model_name)
            except Exception as exc:  # noqa: BLE001 - deliberately broad, see below
                last_error = exc
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                    print(
                        f"[GeminiModelClient] Request failed "
                        f"(attempt {attempt}/{MAX_RETRIES}): {exc}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    # Rebuild the client in case the failure was a dead/stale
                    # pooled connection rather than something retrying on
                    # the same connection would fix.
                    self._client = genai.Client(api_key=self.api_key)
 
        raise last_error
 
    def message_retrieval(self, response: _GeminiResponse):
        return [choice.message.content for choice in response.choices]
 
    def cost(self, response: _GeminiResponse) -> float:
        # Gemini free-tier / no live billing hookup here — return 0.
        return 0.0
 
    @staticmethod
    def get_usage(response: _GeminiResponse) -> dict:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "model": response.model,
        }
 