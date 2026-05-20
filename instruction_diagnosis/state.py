"""State models for Socratic Requirement Refinement."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RefinementStatus = Literal[
    "diagnosing",
    "clarifying",
    "refining",
    "ready",
    "failed",
]


class InstructionRefinementResult(BaseModel):
    """Structured output produced by the requirement refiner."""

    refined_instruction: str = Field(
        description="The clearer, more accurate, more executable instruction."
    )
    remaining_issues: str = Field(
        default="",
        description="Any issues that still remain after refinement.",
    )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "InstructionRefinementResult":
        """Create a result from a dict with Pydantic v1/v2 compatibility."""

        if hasattr(cls, "model_validate"):
            return cls.model_validate(data)  # type: ignore[attr-defined]
        return cls.parse_obj(data)


class RefinementState(BaseModel):
    """Mutable state for the multi-turn requirement refinement loop."""

    original_instruction: str
    current_instruction: str
    refined_instruction: str | None = None
    remaining_issues: str = ""
    diagnosis_history: list[Any] = Field(default_factory=list)
    user_clarifications: list[str] = Field(default_factory=list)
    round_id: int = 0
    max_rounds: int = 3
    status: RefinementStatus = "diagnosing"
