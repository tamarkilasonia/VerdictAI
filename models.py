"""Chat model factory: the only module that knows how we talk to the AI services.

Gemini and Groq both offer an OpenAI-compatible API, so one client
(LangChain's ChatOpenAI via init_chat_model) reaches both: only the base URL,
the key and the model name differ, and those come from .env.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable

from config import ModelSpec

# Cloud models usually answer in seconds; the limit only stops a stuck request
# from freezing the app forever.
REQUEST_TIMEOUT_SECONDS = 60

# Only temporary problems are worth retrying: the service could not be reached
# or asked us to slow down (free tiers allow a few requests per minute).
# Errors such as a wrong key or model name would just fail again.
RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    openai.RateLimitError,
    openai.APIConnectionError,  # also covers timeouts
)


def get_chat_model(spec: ModelSpec, *, temperature: float = 0.7, json_mode: bool = False) -> Runnable:
    """Create a chat model from a ModelSpec (name + base URL + key from .env).

    json_mode: ask the service to reply with a JSON object (used by the auditor).

    The model is wrapped with LangChain's built-in retry, which waits a bit
    longer after each failure (exponential backoff with random jitter).
    """
    options: dict[str, Any] = {}
    if json_mode:
        options["model_kwargs"] = {"response_format": {"type": "json_object"}}
    model = init_chat_model(
        spec.name,
        model_provider="openai",  # the OpenAI *format*, not the OpenAI company
        base_url=spec.base_url,
        api_key=spec.api_key,
        temperature=temperature,
        timeout=REQUEST_TIMEOUT_SECONDS,
        # The client's own hidden retries are switched off so that the one
        # visible retry below is the only one (otherwise they would multiply).
        max_retries=0,
        **options,
    )
    return model.with_retry(
        retry_if_exception_type=RETRYABLE_ERRORS,
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )


def reply_text(reply: Any) -> str:
    """Plain text of a model reply (shared by the answering models and the auditor)."""
    content = reply.content
    # Most providers return a string; some return a list of content parts.
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content.strip()


def friendly_error(error: Exception, spec: ModelSpec) -> str:
    """Turn a service error into a short message for the UI (no stack trace)."""
    name = spec.name
    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return (
            f"The API key for {name!r} was rejected. Check it in .env "
            "(PRIMARY_API_KEY for Gemini, SECONDARY_API_KEY for Groq)."
        )
    if isinstance(error, openai.RateLimitError):
        return (
            f"Free usage limit reached for {name!r}. Wait a minute and try again; "
            "if it keeps happening, today's free limit is used up (it resets daily)."
        )
    if isinstance(error, openai.NotFoundError):
        return f"Model {name!r} was not found. Check the model name in .env."
    if isinstance(error, openai.APITimeoutError):  # before APIConnectionError: it is a subclass
        return f"{name!r} took too long to answer. Please try again."
    if isinstance(error, openai.APIConnectionError):
        return f"Cannot reach {spec.base_url}. Check your internet connection."
    if isinstance(error, openai.BadRequestError):
        return f"{name!r} rejected the request: {error.message}"
    return f"{name!r} failed: {type(error).__name__}: {error}"
