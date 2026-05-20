from __future__ import annotations

from instruction_diagnosis.state import (
    InstructionRefinementResult,
    RefinementState,
)


def test_refinement_state_initializes_defaults() -> None:
    state = RefinementState(
        original_instruction="raw requirement",
        current_instruction="raw requirement",
    )

    assert state.original_instruction == "raw requirement"
    assert state.current_instruction == "raw requirement"
    assert state.refined_instruction is None
    assert state.remaining_issues == ""
    assert state.diagnosis_history == []
    assert state.user_clarifications == []
    assert state.round_id == 0
    assert state.max_rounds == 3
    assert state.status == "diagnosing"


def test_instruction_refinement_result_initializes() -> None:
    result = InstructionRefinementResult(
        refined_instruction="clearer requirement",
        remaining_issues="",
    )

    assert result.refined_instruction == "clearer requirement"
    assert result.remaining_issues == ""
