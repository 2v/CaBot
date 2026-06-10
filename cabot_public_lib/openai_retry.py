"""Retry wrapper for OpenAI API calls.

Transient failures — rate limits, timeouts, dropped connections, 5xx — are retried with
exponential backoff (honoring the server's Retry-After when present). One class of 429 is
deliberately NOT retried: "Request too large", where a single request exceeds the
organization's tokens-per-minute cap for the model. Waiting cannot fix that, so it is
surfaced immediately with guidance instead of a stack trace.
"""
import random
import sys
import time

import openai

MAX_ATTEMPTS = 6
BASE_DELAY = 5.0    # seconds; doubles each attempt
MAX_DELAY = 120.0

RETRYABLE = (openai.RateLimitError, openai.APITimeoutError,
             openai.APIConnectionError, openai.InternalServerError)


class RequestTooLargeError(RuntimeError):
    """A single request exceeds the org's tokens-per-minute limit for the model."""


def _is_request_too_large(err):
    if not isinstance(err, openai.RateLimitError):
        return False
    return "request too large" in str(err).lower()


def _retry_delay(err, attempt):
    response = getattr(err, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), MAX_DELAY)
            except ValueError:
                pass
    return min(BASE_DELAY * (2 ** attempt), MAX_DELAY) * (0.5 + random.random())


def call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying transient OpenAI errors with backoff."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except RETRYABLE as e:
            if _is_request_too_large(e):
                raise RequestTooLargeError(
                    "OpenAI rejected this request as larger than your organization's "
                    "tokens-per-minute (TPM) limit for the model, so retrying cannot "
                    "succeed. Raise the model's TPM tier at "
                    "https://platform.openai.com/account/rate-limits (free/low tiers are "
                    "too small for CaBot's case + exemplar + literature context), or use "
                    "a base model your account has a higher limit for."
                ) from e
            if attempt == MAX_ATTEMPTS - 1:
                raise
            delay = _retry_delay(e, attempt)
            print(f"OpenAI {type(e).__name__}; retrying in {delay:.0f}s "
                  f"(attempt {attempt + 2}/{MAX_ATTEMPTS})", file=sys.stderr, flush=True)
            time.sleep(delay)
