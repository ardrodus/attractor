"""High-level SDK for running Attractor pipelines as library calls.

This module provides the simple, one-call API for embedding Attractor
pipelines inside larger applications (e.g., Dark Factory).

Usage::

    from attractor_pipeline.sdk import execute

    # Bedrock (SSO/profile auth)
    result = await execute(
        "factory/pipelines/dark_forge.dot",
        provider="bedrock",
        aws_profile="pdi-bedrock",
        context={"issue": issue_data},
    )

    # Direct API key
    result = await execute(
        "factory/pipelines/dark_forge.dot",
        provider="anthropic",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        context={"issue": issue_data},
    )

    # With callbacks
    result = await execute(
        "factory/pipelines/crucible.dot",
        provider="bedrock",
        context={"base_sha": "abc123", "head_sha": "def456"},
        on_event=my_event_handler,
        on_human_gate=my_approval_callback,
    )
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from attractor_pipeline.engine.events import PipelineEvent
from attractor_pipeline.engine.runner import PipelineResult, PipelineStatus


@dataclass
class ExecuteConfig:
    """Configuration for pipeline execution."""

    # LLM provider
    provider: str = "bedrock"
    model: str | None = None
    api_key: str | None = None

    # AWS Bedrock settings
    aws_profile: str | None = None
    aws_region: str = "us-east-1"

    # Pipeline context (variables available to nodes via $variable)
    context: dict[str, Any] = field(default_factory=dict)

    # Stylesheet for model assignment per node
    stylesheet_path: str | None = None

    # Execution options
    use_tools: bool = True
    logs_dir: str | None = None

    # Callbacks
    on_event: Callable[[PipelineEvent], None] | None = None
    on_human_gate: Callable[[str, str], str] | None = None


async def execute(
    dotfile: str | Path,
    *,
    provider: str = "bedrock",
    model: str | None = None,
    api_key: str | None = None,
    aws_profile: str | None = None,
    aws_region: str = "us-east-1",
    context: dict[str, Any] | None = None,
    stylesheet_path: str | None = None,
    use_tools: bool = True,
    logs_dir: str | None = None,
    on_event: Callable[[PipelineEvent], None] | None = None,
    on_human_gate: Callable[[str, str], str] | None = None,
) -> PipelineResult:
    """Execute a DOT pipeline and return the result.

    This is the one-call API. Parse, validate, configure LLM client,
    and run — all in one function.

    Args:
        dotfile: Path to the DOT pipeline file.
        provider: LLM provider ('bedrock', 'anthropic', 'openai', 'gemini').
        model: Model ID. Auto-resolved from provider if not specified.
        api_key: API key for non-Bedrock providers. Falls back to env vars.
        aws_profile: AWS profile name for Bedrock (falls back to AWS_PROFILE).
        aws_region: AWS region for Bedrock (default: us-east-1).
        context: Initial pipeline context (variables for $expansion).
        stylesheet_path: Path to a .styles file for per-node model assignment.
        use_tools: Whether agent nodes get developer tools (default: True).
        logs_dir: Directory for logs and checkpoints.
        on_event: Callback invoked for each pipeline event.
        on_human_gate: Callback for human gate nodes. Receives (node_id, prompt),
            returns the human's answer. If None, human gates auto-approve.

    Returns:
        PipelineResult with status, context, completed nodes, and any error.

    Raises:
        FileNotFoundError: If dotfile doesn't exist.
        ValueError: If the DOT file fails validation.
        ConfigurationError: If LLM provider can't be configured.
    """
    from attractor_llm.client import Client
    from attractor_pipeline import (
        HandlerRegistry,
        parse_dot,
        register_default_handlers,
    )
    from attractor_pipeline import run_pipeline as _run_pipeline
    from attractor_pipeline.backends import AgentLoopBackend, DirectLLMBackend
    from attractor_pipeline.validation import validate_or_raise

    # --- Parse ---
    path = Path(dotfile)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {dotfile}")

    source = path.read_text(encoding="utf-8")
    graph = parse_dot(source)

    # --- Validate ---
    validate_or_raise(graph)

    # --- Configure LLM Client ---
    client = Client()
    resolved_model = model or _default_model_for_provider(provider)
    _register_provider(client, provider, api_key, aws_profile, aws_region)

    # --- Apply Stylesheet ---
    if stylesheet_path:
        from attractor_pipeline.stylesheet import apply_stylesheet, parse_stylesheet

        style_path = Path(stylesheet_path)
        if style_path.exists():
            stylesheet = parse_stylesheet(style_path.read_text(encoding="utf-8"))
            apply_stylesheet(graph, stylesheet)

    # --- Set up Backend ---
    if use_tools:
        backend = AgentLoopBackend(
            client, default_model=resolved_model, default_provider=provider
        )
    else:
        backend = DirectLLMBackend(
            client, default_model=resolved_model, default_provider=provider
        )

    # --- Set up Handlers ---
    registry = HandlerRegistry()
    register_default_handlers(registry, codergen_backend=backend)

    # --- Human Gate Callback ---
    if on_human_gate:
        from attractor_pipeline.handlers import CallbackInterviewer, HumanHandler

        interviewer = CallbackInterviewer(
            callback=lambda q: on_human_gate(q.node_id or "", q.text)
        )
        registry.register("human", HumanHandler(interviewer=interviewer))

    # --- Logs ---
    logs_root = None
    if logs_dir:
        logs_root = Path(logs_dir)
        logs_root.mkdir(parents=True, exist_ok=True)

    # --- Execute ---
    async with client:
        result = await _run_pipeline(
            graph,
            registry,
            context=context,
            logs_root=logs_root,
            on_event=on_event,
        )

    return result


def _default_model_for_provider(provider: str) -> str:
    """Return the default model for a provider."""
    defaults = {
        "bedrock": "us.anthropic.claude-sonnet-4-5-v2-0",
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-5.2",
        "gemini": "gemini-3-flash-preview",
    }
    return defaults.get(provider, "claude-sonnet-4-5")


def _register_provider(
    client: Client,
    provider: str,
    api_key: str | None,
    aws_profile: str | None,
    aws_region: str,
) -> None:
    """Register the appropriate adapter on the client."""
    if provider == "bedrock":
        from attractor_llm.adapters.bedrock import BedrockAdapter, BedrockConfig

        profile = aws_profile or os.environ.get("AWS_PROFILE")
        region = aws_region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        config = BedrockConfig(region=region, profile_name=profile)
        client.register_adapter("bedrock", BedrockAdapter(config))

    elif provider == "anthropic":
        from attractor_llm.adapters.anthropic import AnthropicAdapter
        from attractor_llm.adapters.base import ProviderConfig

        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        client.register_adapter("anthropic", AnthropicAdapter(ProviderConfig(api_key=key)))

    elif provider == "openai":
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.openai import OpenAIAdapter

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        client.register_adapter("openai", OpenAIAdapter(ProviderConfig(api_key=key)))

    elif provider == "gemini":
        from attractor_llm.adapters.base import ProviderConfig
        from attractor_llm.adapters.gemini import GeminiAdapter

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        client.register_adapter("gemini", GeminiAdapter(ProviderConfig(api_key=key)))

    else:
        from attractor_llm.errors import ConfigurationError

        raise ConfigurationError(f"Unknown provider: {provider!r}")
