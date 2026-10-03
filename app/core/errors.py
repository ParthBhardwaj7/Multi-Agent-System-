"""Domain error taxonomy.

Every error carries a machine-readable code, an HTTP status, and a
client-safe message. Internal detail (stack traces, provider payloads) is
logged, never returned to clients.
"""

from typing import Any


class AppError(Exception):
    """Base class for all expected application errors."""

    status_code = 500
    code = "internal_error"
    message = "An internal error occurred."
    retryable = False

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        if retryable is not None:
            self.retryable = retryable
        super().__init__(self.message)


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
    message = "Request validation failed."
    retryable = False


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_error"
    message = "Authentication required or invalid."
    retryable = False


class AuthorizationError(AppError):
    status_code = 403
    code = "authorization_error"
    message = "You do not have permission to perform this action."
    retryable = False


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Resource not found."
    retryable = False


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "The request conflicts with current state."
    retryable = True


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limit_exceeded"
    message = "Rate limit exceeded. Retry later."
    retryable = True

    def __init__(self, message: str | None = None, *, retry_after: int = 1, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = max(1, retry_after)


class LLMError(AppError):
    status_code = 502
    code = "llm_error"
    message = "The language model request failed."
    retryable = True


class LLMTimeoutError(LLMError):
    code = "llm_timeout"
    message = "The language model request timed out."
    retryable = True


class LLMRateLimitError(LLMError):
    code = "llm_rate_limited"
    message = "The language model provider is rate limiting requests."
    retryable = True


class LLMResponseError(LLMError):
    """LLM returned output that failed schema validation."""

    code = "llm_invalid_response"
    message = "The language model returned an invalid response."
    retryable = False


class ToolError(AppError):
    status_code = 502
    code = "tool_error"
    message = "A tool execution failed."
    retryable = True


class ToolUnavailableError(ToolError):
    code = "tool_unavailable"
    message = "The requested tool is unavailable."
    retryable = False


class ToolAuthorizationError(ToolError):
    status_code = 403
    code = "tool_authorization_error"
    message = "Tool use is not permitted for this principal."
    retryable = False


class ToolValidationError(ToolError):
    code = "tool_invalid_input"
    message = "Tool input failed validation."
    retryable = False


class DatabaseError(AppError):
    status_code = 503
    code = "database_error"
    message = "A database operation failed."
    retryable = True


class PromptInjectionError(AppError):
    """Raised when policy checks detect a likely injection attempt in tool input."""

    status_code = 400
    code = "policy_violation"
    message = "Request rejected by content policy."
    retryable = False
