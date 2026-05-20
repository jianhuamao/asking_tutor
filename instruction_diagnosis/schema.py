"""Structured schema for Instruction Diagnosis results."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field


IssueType = Literal[
    "none",
    "ambiguity",
    "missing_constraint",
    "logical_contradiction",
    "conceptual_confusion",
    "false_premise",
    "too_broad",
    "infeasible",
    "underspecified",
]

Action = Literal["answer", "clarify", "revise"]


class InstructionDiagnosisResult(BaseModel):
    """Final public output for the pre-answer diagnostic agent."""

    is_clear: bool = Field(description="Whether the instruction is clear enough.")
    is_accurate: bool = Field(description="Whether concepts and premises are accurate.")
    is_logically_consistent: bool = Field(
        description="Whether the instruction is internally consistent."
    )
    has_missing_constraints: bool = Field(
        description="Whether critical constraints are missing."
    )
    has_ambiguity: bool = Field(description="Whether obvious ambiguity exists.")
    issue_type: IssueType = Field(description="Primary issue category.")
    explanation: str = Field(description="Friendly explanation of the diagnosis.")
    revised_instruction: str = Field(
        description="A clearer rewritten instruction, or an empty string."
    )
    clarification_question: str = Field(
        description="One most important clarification question, or an empty string."
    )
    action: Action = Field(description="Recommended next action.")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "InstructionDiagnosisResult":
        """Create a result from a dict with Pydantic v1/v2 compatibility."""

        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)  # type: ignore[attr-defined]
        return cls.parse_obj(data)

    @classmethod
    def from_json(cls, value: str) -> "InstructionDiagnosisResult":
        """Parse a JSON object, tolerating fenced model output."""

        json_text = extract_json_object(value)
        data = json.loads(json_text)
        if not isinstance(data, dict):
            raise ValueError("Instruction diagnosis output must be a JSON object.")
        return cls.from_mapping(data)

    def to_pretty_json(self) -> str:
        """Serialize as stable, pretty JSON for CLI output."""

        if hasattr(self, "model_dump_json"):
            return self.model_dump_json(indent=2)  # type: ignore[attr-defined]
        return self.json(indent=2, ensure_ascii=False)


def extract_json_object(value: str) -> str:
    """Extract the first balanced JSON object from raw LLM text.

    DSPy normally returns only the output field, but this keeps the parser usable
    when a model wraps JSON in Markdown fences or a short preface.
    """

    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in instruction diagnosis output.")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Unbalanced JSON object in instruction diagnosis output.")
