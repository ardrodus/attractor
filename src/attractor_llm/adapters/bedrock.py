"""AWS Bedrock adapter for Anthropic models.

Routes requests through AWS Bedrock Runtime using boto3 with SigV4 auth.
Reuses the Anthropic request/response translation since Bedrock's
invoke_model API accepts the Anthropic Messages API format directly.

Requires: boto3 (optional dependency)

Auth resolution order:
1. Explicit profile_name in BedrockConfig
2. AWS_PROFILE environment variable
3. SSO session credentials (~/.aws/sso/cache/)
4. Standard boto3 credential chain (env vars, instance role, etc.)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from attractor_llm.errors import (
    ConfigurationError,
    classify_http_error,
)
from attractor_llm.types import (
    ContentPart,
    ContentPartKind,
    FinishReason,
    Message,
    Request,
    Response,
    Role,
    StreamEvent,
    StreamEventKind,
    Usage,
)

from .base import ProviderConfig

# Default Bedrock region
DEFAULT_REGION = "us-east-1"

# Model ID mapping: short alias -> Bedrock model ID
BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1-0",
    "claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6-v1-0",
    "claude-sonnet-4-5": "us.anthropic.claude-sonnet-4-5-v2-0",
    "claude-haiku-4-5": "us.anthropic.claude-haiku-4-5-v1-0",
}

DEFAULT_MAX_TOKENS = 8192


@dataclass(frozen=True)
class BedrockConfig:
    """Configuration specific to AWS Bedrock.

    Extends the base ProviderConfig with AWS-specific settings.
    """

    region: str = DEFAULT_REGION
    profile_name: str | None = None
    timeout: float = 120.0
    default_headers: dict[str, str] = field(default_factory=dict)


class BedrockAdapter:
    """AWS Bedrock adapter using boto3.

    Implements ProviderAdapter protocol. Translates unified Request/Response
    to Anthropic Messages API format, then routes through Bedrock Runtime.
    """

    def __init__(self, config: BedrockConfig | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ConfigurationError(
                "boto3 is required for Bedrock. Install with: pip install boto3"
            ) from exc

        self._config = config or BedrockConfig()
        session_kwargs: dict[str, Any] = {}
        if self._config.profile_name:
            session_kwargs["profile_name"] = self._config.profile_name

        session = boto3.Session(**session_kwargs)
        self._client = session.client(
            "bedrock-runtime",
            region_name=self._config.region,
        )

    @property
    def provider_name(self) -> str:
        return "bedrock"

    def _resolve_model_id(self, model: str) -> str:
        """Resolve a short model name to a Bedrock model ID.

        If the model already looks like a Bedrock ID (contains dots with
        'anthropic' prefix), use it as-is. Otherwise, look up in the map.
        """
        if "anthropic" in model and "." in model:
            return model  # Already a Bedrock model ID
        return BEDROCK_MODEL_MAP.get(model, model)

    # ------------------------------------------------------------------ #
    # Request translation (Unified -> Anthropic Messages API for Bedrock)
    # ------------------------------------------------------------------ #

    def _translate_request(self, request: Request) -> dict[str, Any]:
        """Translate unified Request to Anthropic Messages API body for Bedrock."""
        messages = request.effective_messages()
        system_parts, conversation = self._split_system(messages)
        anthropic_messages = self._translate_messages(conversation)
        anthropic_messages = self._enforce_alternation(anthropic_messages)

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or DEFAULT_MAX_TOKENS,
        }

        if system_parts:
            body["system"] = system_parts

        if request.tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters or {"type": "object", "properties": {}},
                }
                for t in request.tools
            ]

        if request.tool_choice:
            if request.tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif request.tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "none":
                body.pop("tools", None)
            else:
                body["tool_choice"] = {"type": "tool", "name": request.tool_choice}

        if request.temperature is not None:
            body["temperature"] = request.temperature

        if request.top_p is not None:
            body["top_p"] = request.top_p

        if request.stop:
            body["stop_sequences"] = request.stop

        # Extended thinking
        anthropic_opts = (request.provider_options or {}).get("bedrock", {})
        if request.reasoning_effort or anthropic_opts.get("thinking"):
            thinking_config = anthropic_opts.get("thinking", {})
            budget = thinking_config.get(
                "budget_tokens", self._thinking_budget(request.reasoning_effort)
            )
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body.pop("temperature", None)

        return body

    def _split_system(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], list[Message]]:
        """Extract system messages into top-level system field."""
        system_parts: list[dict[str, Any]] = []
        conversation: list[Message] = []

        for msg in messages:
            if msg.role in (Role.SYSTEM, Role.DEVELOPER):
                for part in msg.content:
                    if part.kind == ContentPartKind.TEXT and part.text:
                        system_parts.append({"type": "text", "text": part.text})
            else:
                conversation.append(msg)

        return system_parts, conversation

    def _translate_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Translate unified messages to Anthropic format."""
        result: list[dict[str, Any]] = []

        for msg in messages:
            role = "user" if msg.role in (Role.USER, Role.TOOL) else "assistant"
            content: list[dict[str, Any]] = []

            for part in msg.content:
                translated = self._translate_content_part(part)
                if translated:
                    content.append(translated)

            if content:
                result.append({"role": role, "content": content})

        return result

    def _translate_content_part(self, part: ContentPart) -> dict[str, Any] | None:
        """Translate a single content part."""
        match part.kind:
            case ContentPartKind.TEXT:
                return {"type": "text", "text": part.text or ""}
            case ContentPartKind.TOOL_CALL:
                import uuid

                args = part.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                return {
                    "type": "tool_use",
                    "id": part.tool_call_id or str(uuid.uuid4()),
                    "name": part.name or "",
                    "input": args or {},
                }
            case ContentPartKind.TOOL_RESULT:
                result: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": part.tool_call_id or "",
                    "content": part.output or "",
                }
                if part.is_error:
                    result["is_error"] = True
                return result
            case ContentPartKind.THINKING:
                return {
                    "type": "thinking",
                    "thinking": part.text or "",
                    **({"signature": part.signature} if part.signature else {}),
                }
            case _:
                return {"type": "text", "text": part.text or ""}

    def _enforce_alternation(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge consecutive same-role messages for strict alternation."""
        if not messages:
            return messages

        merged: list[dict[str, Any]] = [
            {**messages[0], "content": list(messages[0]["content"])}
        ]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                merged[-1]["content"].extend(msg["content"])
            else:
                merged.append(msg)

        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": [{"type": "text", "text": "..."}]})

        return merged

    def _thinking_budget(self, effort: str | None) -> int:
        """Map reasoning_effort to thinking budget tokens."""
        match effort:
            case "low":
                return 2048
            case "medium":
                return 8192
            case "high":
                return 32768
            case _:
                return 8192

    # ------------------------------------------------------------------ #
    # Response translation (Bedrock -> Unified)
    # ------------------------------------------------------------------ #

    def _translate_response(self, data: dict[str, Any], request: Request) -> Response:
        """Translate Bedrock/Anthropic response to unified Response."""
        content_parts: list[ContentPart] = []

        for block in data.get("content", []):
            part = self._translate_response_block(block)
            if part:
                content_parts.append(part)

        stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = self._map_finish_reason(stop_reason)

        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
        )

        return Response(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            provider="bedrock",
            message=Message(role=Role.ASSISTANT, content=content_parts),
            finish_reason=finish_reason,
            usage=usage,
            raw=data,
        )

    def _translate_response_block(self, block: dict[str, Any]) -> ContentPart | None:
        """Translate a single response block."""
        block_type = block.get("type")

        match block_type:
            case "text":
                return ContentPart.text_part(block.get("text", ""))
            case "tool_use":
                args = block.get("input", {})
                return ContentPart.tool_call_part(
                    tool_call_id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=args if isinstance(args, dict) else str(args),
                )
            case "thinking":
                return ContentPart.thinking_part(
                    text=block.get("thinking", ""),
                    signature=block.get("signature"),
                )
            case _:
                return None

    def _map_finish_reason(self, stop_reason: str) -> FinishReason:
        """Map stop_reason to unified FinishReason."""
        match stop_reason:
            case "end_turn" | "stop_sequence":
                return FinishReason.STOP
            case "tool_use":
                return FinishReason.TOOL_CALLS
            case "max_tokens":
                return FinishReason.MAX_TOKENS
            case _:
                return FinishReason.STOP

    # ------------------------------------------------------------------ #
    # complete() -- blocking call via boto3
    # ------------------------------------------------------------------ #

    async def complete(self, request: Request) -> Response:
        """Send a request via Bedrock invoke_model."""
        import anyio

        model_id = self._resolve_model_id(request.model)
        body = self._translate_request(request)

        def _invoke() -> dict[str, Any]:
            response = self._client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            return json.loads(response["body"].read())

        # Run boto3 sync call in a thread to avoid blocking the event loop
        data = await anyio.to_thread.run_sync(_invoke)
        return self._translate_response(data, request)

    # ------------------------------------------------------------------ #
    # stream() -- SSE streaming via boto3
    # ------------------------------------------------------------------ #

    async def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        """Send a request and yield streaming events via Bedrock."""
        import anyio

        model_id = self._resolve_model_id(request.model)
        body = self._translate_request(request)

        def _invoke_stream() -> Any:
            response = self._client.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            return response

        response = await anyio.to_thread.run_sync(_invoke_stream)

        # Yield START event
        yield StreamEvent(
            kind=StreamEventKind.START,
            model=model_id,
            provider="bedrock",
        )

        # Process Bedrock's event stream
        event_stream = response.get("body", [])
        for event in event_stream:
            chunk = event.get("chunk")
            if not chunk:
                continue

            data = json.loads(chunk["bytes"])
            event_type = data.get("type", "")

            match event_type:
                case "content_block_start":
                    cb = data.get("content_block", {})
                    if cb.get("type") == "text":
                        yield StreamEvent(kind=StreamEventKind.TEXT_START)
                    elif cb.get("type") == "tool_use":
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_START,
                            tool_call_id=cb.get("id", ""),
                            tool_name=cb.get("name", ""),
                        )

                case "content_block_delta":
                    delta = data.get("delta", {})
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.TEXT_DELTA,
                            text=delta.get("text", ""),
                        )
                    elif delta_type == "thinking_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.THINKING_DELTA,
                            text=delta.get("thinking", ""),
                        )
                    elif delta_type == "input_json_delta":
                        yield StreamEvent(
                            kind=StreamEventKind.TOOL_CALL_DELTA,
                            arguments_delta=delta.get("partial_json", ""),
                        )

                case "content_block_stop":
                    pass  # Block complete

                case "message_delta":
                    delta = data.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    if stop_reason:
                        yield StreamEvent(
                            kind=StreamEventKind.FINISH,
                            finish_reason=self._map_finish_reason(stop_reason),
                        )
                    usage_data = data.get("usage", {})
                    if usage_data:
                        yield StreamEvent(
                            kind=StreamEventKind.USAGE,
                            usage=Usage(
                                output_tokens=usage_data.get("output_tokens", 0),
                            ),
                        )

                case "message_stop":
                    pass

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """No persistent connections to close with boto3."""
        pass
