"""Small serializable records used across all gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RawPrompt:
    source_dataset: str
    source_id: str
    task_family: str
    prompt: str
    reference_answer: str
    verifier_type: str
    split: str = ""
    documents: list[dict[str, str]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolForgeEpisode:
    episode_id: str
    source_dataset: str
    source_id: str
    task_family: str
    difficulty: str
    prompt: str
    tools: list[str]
    messages: list[dict]
    reference_answer: str
    final_answer: str | None
    verifier_type: str
    tool_calls: list[dict]
    tool_call_count: int
    answer_correct: bool
    schema_valid: bool
    execution_success: bool
    generator_model: str
    sampling_hint: str
    split: str

    def to_dict(self) -> dict:
        return asdict(self)
