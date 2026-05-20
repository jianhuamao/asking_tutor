from __future__ import annotations

from types import SimpleNamespace

from instruction_diagnosis.refiner import InstructionRefiner
from instruction_diagnosis.state import InstructionRefinementResult


def test_refiner_returns_pydantic_result_from_fake_prediction() -> None:
    def fake_predictor(**kwargs: str) -> SimpleNamespace:
        assert kwargs["original_instruction"] == "原始需求"
        assert kwargs["current_instruction"] == "当前需求"
        assert kwargs["diagnosis_explanation"] == "缺少关键约束"
        assert kwargs["user_clarification"] == "使用小数据集"
        return SimpleNamespace(
            refined_instruction="请基于小数据集设计一个可执行实验。",
            remaining_issues="",
        )

    result = InstructionRefiner(predictor=fake_predictor).refine(
        original_instruction="原始需求",
        current_instruction="当前需求",
        diagnosis_explanation="缺少关键约束",
        user_clarification="使用小数据集",
    )

    assert isinstance(result, InstructionRefinementResult)
    assert result.refined_instruction == "请基于小数据集设计一个可执行实验。"
    assert result.remaining_issues == ""


def test_refiner_fallback_merges_current_instruction_and_user_clarification() -> None:
    result = InstructionRefiner(
        predictor=lambda **_: SimpleNamespace(
            refined_instruction="",
            remaining_issues="",
        )
    ).refine(
        original_instruction="原始需求",
        current_instruction="当前需求",
        diagnosis_explanation="缺少关键约束",
        user_clarification="补充一个约束",
    )

    assert "当前需求" in result.refined_instruction
    assert "补充一个约束" in result.refined_instruction
    assert result.remaining_issues == "The model did not provide explicit remaining issues."


def test_refiner_accepts_dict_prediction() -> None:
    result = InstructionRefiner(
        predictor=lambda **_: {
            "refined_instruction": "更清晰的需求",
            "remaining_issues": "仍缺少指标",
        }
    ).refine(
        original_instruction="原始需求",
        current_instruction="当前需求",
        diagnosis_explanation="缺少指标",
        user_clarification="先用 accuracy",
    )

    assert result.refined_instruction == "更清晰的需求"
    assert result.remaining_issues == "仍缺少指标"
