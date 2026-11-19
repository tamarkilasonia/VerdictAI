"""Chat model factory: the only module that knows which provider we talk to.

Every other module asks for a model by name and gets back something it can
`.invoke(messages)`. Switching provider means editing .env, not this code.
"""

from __future__ import annotations

from typing import Any

import httpx  # the HTTP client used by the Ollama client; installed with langchain-ollama
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable

from config import Settings, get_settings

# A local 3B model can be slow on a laptop, so allow a generous time limit.
# Without one, a stuck server would freeze the app forever.
REQUEST_TIMEOUT_SECONDS = 120

# Only temporary problems are worth retrying: the server could not be reached
# or is rate-limiting us. Errors such as "model not found" would just fail
# again. A slow answer (read timeout) is not retried either: a slow local
# model would only be slow again, tripling the wait.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,  # the Ollama client raises this when the server is down
    httpx.ConnectError,
    httpx.ConnectTimeout,
)
try:  # The openai package is only present if you switch to an OpenAI-style provider.
    import openai

    RETRYABLE_ERRORS += (openai.RateLimitError, openai.APIConnectionError)
except ImportError:
    pass


def _provider_options(settings: Settings, json_schema: dict | None) -> dict[str, Any]:
    """Translate our generic options into the parameter names each provider expects."""
    if settings.model_provider == "ollama":
        options: dict[str, Any] = {
            "base_url": settings.model_base_url,
            "client_kwargs": {"timeout": REQUEST_TIMEOUT_SECONDS},
        }
        if json_schema is not None:
            # Ollama can force the output to match a JSON schema. This is much
            # more reliable for small models than only asking for JSON in words.
            options["format"] = json_schema
        return options

    # Other providers (e.g. "openai" or an OpenAI-compatible host).
    options = {"timeout": REQUEST_TIMEOUT_SECONDS}
    if settings.model_base_url:
        options["base_url"] = settings.model_base_url
    if settings.model_api_key:
        options["api_key"] = settings.model_api_key
    if json_schema is not None and settings.model_provider == "openai":
        options["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return options


def get_chat_model(
    model_name: str,
    *,
    temperature: float = 0.7,
    json_schema: dict | None = None,
) -> Runnable:
    """Create a chat model for `model_name`, configured from .env.

    json_schema: if given, ask the provider to return JSON (matching the schema
    where the provider supports it). Used by the auditor.

    The model is wrapped with LangChain's built-in retry, which waits a bit
    longer after each failure (exponential backoff with random jitter).
    """
    settings = get_settings()
    model = init_chat_model(
        model_name,
        model_provider=settings.model_provider,
        temperature=temperature,
        **_provider_options(settings, json_schema),
    )
    return model.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )


def friendly_error(error: Exception, model_name: str = "") -> str:
    """Turn a provider exception into a short message for the UI (no stack trace)."""
    settings = get_settings()
    if isinstance(error, (ConnectionError, httpx.ConnectError)):
        return (
            f"Cannot reach the model server at {settings.model_base_url}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return f"The model {model_name!r} took too long to answer. Please try again."
    # Ollama answers HTTP 404 when the model has not been downloaded yet.
    if getattr(error, "status_code", None) == 404 and settings.model_provider == "ollama":
        return f"Model {model_name!r} is not installed. Run: ollama pull {model_name}"
    return f"Model {model_name!r} failed: {type(error).__name__}: {error}"
