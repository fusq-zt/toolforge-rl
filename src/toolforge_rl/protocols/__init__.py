"""Model/tool wire protocols."""

from .toolstar import (
    OFFICIAL_SYSTEM_PROMPT,
    Action,
    ActionKind,
    ProtocolValidation,
    ToolNameAdapter,
    extract_final_answer,
    parse_next_action,
    render_result,
    trim_at_first_stop,
    validate_transcript,
)

__all__ = [
    "OFFICIAL_SYSTEM_PROMPT",
    "Action",
    "ActionKind",
    "ProtocolValidation",
    "ToolNameAdapter",
    "extract_final_answer",
    "parse_next_action",
    "render_result",
    "trim_at_first_stop",
    "validate_transcript",
]
