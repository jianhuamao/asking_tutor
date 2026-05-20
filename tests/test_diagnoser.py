from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from instruction_diagnosis import InstructionDiagnoser, InstructionDiagnosisResult
from instruction_diagnosis.schema import extract_json_object


def test_schema_accepts_expected_clear_example() -> None:
    result = InstructionDiagnosisResult(
        is_clear=True,
        is_accurate=True,
        is_logically_consistent=True,
        has_missing_constraints=False,
        has_ambiguity=False,
        issue_type="none",
        explanation="该问题目标明确，可以直接解释两个算法在 non-IID 场景下的区别。",
        revised_instruction="",
        clarification_question="",
        action="answer",
    )

    assert result.action == "answer"
    assert result.issue_type == "none"


def test_schema_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValidationError):
        InstructionDiagnosisResult(
            is_clear=True,
            is_accurate=True,
            is_logically_consistent=True,
            has_missing_constraints=False,
            has_ambiguity=False,
            issue_type="unclear",
            explanation="bad enum",
            revised_instruction="",
            clarification_question="",
            action="ask",
        )


def test_diagnoser_parses_fake_dspy_json_output() -> None:
    payload = {
        "is_clear": False,
        "is_accurate": True,
        "is_logically_consistent": True,
        "has_missing_constraints": True,
        "has_ambiguity": False,
        "issue_type": "missing_constraint",
        "explanation": "实验目标可理解，但缺少任务、数据集、baseline 或指标等关键约束。",
        "revised_instruction": "",
        "clarification_question": "你最希望在哪个任务和数据集上证明方法优于哪个 baseline？",
        "action": "clarify",
    }

    def fake_predictor(**kwargs: str) -> SimpleNamespace:
        assert kwargs["user_instruction"] == "帮我设计一个实验，证明我的方法比 baseline 好。"
        return SimpleNamespace(diagnosis_json=json.dumps(payload, ensure_ascii=False))

    result = InstructionDiagnoser(predictor=fake_predictor).diagnose(
        "帮我设计一个实验，证明我的方法比 baseline 好。"
    )

    assert result.action == "clarify"
    assert result.issue_type == "missing_constraint"
    assert result.has_missing_constraints is True
    assert result.clarification_question


def test_diagnoser_accepts_direct_field_prediction_for_logical_contradiction() -> None:
    fake_prediction = SimpleNamespace(
        is_clear=False,
        is_accurate=False,
        is_logically_consistent=False,
        has_missing_constraints=True,
        has_ambiguity=False,
        issue_type="logical_contradiction",
        explanation=(
            "该需求同时要求完全无监督且绝对准确，但没有人工标注或外部参照时，"
            "很难定义并验证绝对准确。"
        ),
        revised_instruction=(
            "我想设计一个尽量减少人工标注的 LLM 自动评测系统，并使用少量 "
            "gold set 校准评测结果。"
        ),
        clarification_question="你是否允许使用少量人工标注的验证集来校准自动评测器？",
        action="revise",
    )

    result = InstructionDiagnoser(predictor=lambda **_: fake_prediction).diagnose(
        "我想做一个完全无监督、完全不依赖人工标注、同时能保证评测结果绝对准确的 LLM 评测系统。"
    )

    assert result.action == "revise"
    assert result.issue_type == "logical_contradiction"
    assert result.is_logically_consistent is False


@pytest.mark.parametrize(
    ("instruction", "issue_type", "action"),
    [
        ("帮我写一篇能中 NeurIPS 的论文。", "too_broad", "clarify"),
        (
            "GEPA 没有训练参数，为什么它的模型权重会随着 rollout 数量增加而更新？",
            "conceptual_confusion",
            "revise",
        ),
    ],
)
def test_research_examples_can_be_represented(
    instruction: str, issue_type: str, action: str
) -> None:
    payload = {
        "is_clear": False,
        "is_accurate": issue_type != "conceptual_confusion",
        "is_logically_consistent": True,
        "has_missing_constraints": issue_type == "too_broad",
        "has_ambiguity": False,
        "issue_type": issue_type,
        "explanation": f"{instruction} 存在 {issue_type} 问题。",
        "revised_instruction": "请将目标拆成一个更小、更可执行的研究任务。",
        "clarification_question": "你想优先完成论文中的哪一部分？"
        if issue_type == "too_broad"
        else "",
        "action": action,
    }

    result = InstructionDiagnoser(predictor=lambda **_: payload).diagnose(instruction)

    assert result.issue_type == issue_type
    assert result.action == action


def test_empty_instruction_returns_underspecified_without_llm_call() -> None:
    def fail_predictor(**_: str) -> None:
        raise AssertionError("predictor should not be called for empty input")

    result = InstructionDiagnoser(predictor=fail_predictor).diagnose("   ")

    assert result.action == "clarify"
    assert result.issue_type == "underspecified"
    assert result.clarification_question


def test_extract_json_object_tolerates_markdown_fences() -> None:
    raw = """```json
{"action": "answer", "issue_type": "none"}
```"""

    assert extract_json_object(raw) == '{"action": "answer", "issue_type": "none"}'
