from __future__ import annotations

import pytest

from instruction_diagnosis.controller import RequirementRefinementController
from instruction_diagnosis.schema import InstructionDiagnosisResult
from instruction_diagnosis.state import InstructionRefinementResult, RefinementState
from instruction_diagnosis import cli
from instruction_diagnosis.interactive_cli import _print_clarification_prompt


def make_diagnosis(
    action: str,
    *,
    explanation: str = "",
    clarification_question: str = "",
    revised_instruction: str = "",
) -> InstructionDiagnosisResult:
    return InstructionDiagnosisResult(
        is_clear=action == "answer",
        is_accurate=True,
        is_logically_consistent=True,
        has_missing_constraints=action != "answer",
        has_ambiguity=False,
        issue_type="none" if action == "answer" else "missing_constraint",
        explanation=explanation,
        revised_instruction=revised_instruction,
        clarification_question=clarification_question,
        action=action,  # type: ignore[arg-type]
    )


class FakeDiagnoser:
    def __init__(self, diagnoses: list[InstructionDiagnosisResult]) -> None:
        self.diagnoses = diagnoses
        self.inputs: list[str] = []

    def diagnose(self, user_instruction: str) -> InstructionDiagnosisResult:
        self.inputs.append(user_instruction)
        if not self.diagnoses:
            raise AssertionError("No fake diagnosis left.")
        return self.diagnoses.pop(0)


class FakeRefiner:
    def __init__(self, refined_instruction: str, remaining_issues: str = "") -> None:
        self.refined_instruction = refined_instruction
        self.remaining_issues = remaining_issues
        self.calls: list[dict[str, str]] = []

    def refine(
        self,
        original_instruction: str,
        current_instruction: str,
        diagnosis_explanation: str,
        user_clarification: str,
    ) -> InstructionRefinementResult:
        self.calls.append(
            {
                "original_instruction": original_instruction,
                "current_instruction": current_instruction,
                "diagnosis_explanation": diagnosis_explanation,
                "user_clarification": user_clarification,
            }
        )
        return InstructionRefinementResult(
            refined_instruction=self.refined_instruction,
            remaining_issues=self.remaining_issues,
        )


def test_start_enters_ready_when_diagnosis_action_is_answer() -> None:
    diagnoser = FakeDiagnoser([make_diagnosis("answer")])
    controller = RequirementRefinementController(
        diagnoser=diagnoser,
        refiner=FakeRefiner("unused"),
    )

    state = controller.start("清晰需求")

    assert state.status == "ready"
    assert state.refined_instruction == "清晰需求"
    assert len(state.diagnosis_history) == 1
    assert diagnoser.inputs == ["清晰需求"]


def test_start_enters_clarifying_when_diagnosis_action_is_clarify() -> None:
    diagnoser = FakeDiagnoser(
        [
            make_diagnosis(
                "clarify",
                explanation="缺少数据集",
                clarification_question="使用哪个数据集？",
            )
        ]
    )
    controller = RequirementRefinementController(
        diagnoser=diagnoser,
        refiner=FakeRefiner("unused"),
    )

    state = controller.start("帮我设计实验")

    assert state.status == "clarifying"
    assert state.refined_instruction is None
    assert controller.get_latest_diagnosis(state).clarification_question == "使用哪个数据集？"


def test_process_user_clarification_updates_state_and_rediagnoses() -> None:
    diagnoser = FakeDiagnoser(
        [
            make_diagnosis("clarify", explanation="缺少数据集"),
            make_diagnosis("answer"),
        ]
    )
    refiner = FakeRefiner("请在 MNIST 上设计实验。")
    controller = RequirementRefinementController(diagnoser=diagnoser, refiner=refiner)

    state = controller.start("帮我设计实验")
    state = controller.process_user_clarification(state, "使用 MNIST")

    assert state.status == "ready"
    assert state.user_clarifications == ["使用 MNIST"]
    assert state.round_id == 1
    assert state.current_instruction == "请在 MNIST 上设计实验。"
    assert state.refined_instruction == "请在 MNIST 上设计实验。"
    assert diagnoser.inputs == ["帮我设计实验", "请在 MNIST 上设计实验。"]
    assert refiner.calls[0]["diagnosis_explanation"] == "缺少数据集"
    assert refiner.calls[0]["user_clarification"] == "使用 MNIST"


def test_process_user_clarification_enters_failed_at_max_rounds() -> None:
    diagnoser = FakeDiagnoser(
        [
            make_diagnosis("clarify", explanation="缺少约束"),
            make_diagnosis("clarify", explanation="仍缺少约束"),
        ]
    )
    controller = RequirementRefinementController(
        diagnoser=diagnoser,
        refiner=FakeRefiner("仍然不够清晰的需求"),
    )

    state = controller.start("过大的需求", max_rounds=1)
    state = controller.process_user_clarification(state, "补充一点信息")

    assert state.status == "failed"
    assert state.round_id == 1
    assert len(state.diagnosis_history) == 2


def test_remaining_issues_prevent_premature_ready_for_fedraa_experiment() -> None:
    diagnoser = FakeDiagnoser(
        [
            make_diagnosis(
                "clarify",
                explanation="实验设计缺少方法、baseline、指标或设置等关键约束。",
                clarification_question="你要证明的具体方法和数据集是什么？",
            ),
            make_diagnosis(
                "answer",
                explanation="需求看起来可以回答。",
                clarification_question="",
            ),
        ]
    )
    refiner = FakeRefiner(
        refined_instruction=(
            "请设计一个实验，证明 FedRAA 在 CIFAR-10 上比其他模型更好。"
        ),
        remaining_issues=(
            "Missing method details, baseline, metrics, and experimental setup."
        ),
    )
    controller = RequirementRefinementController(diagnoser=diagnoser, refiner=refiner)

    state = controller.start("帮我设计一个实验，证明我的方法比 baseline 好。")
    state = controller.process_user_clarification(
        state,
        "我想要证明 FedRAA，在 CIFAR-10 上比其他模型更好。",
    )

    assert state.status == "clarifying"
    assert state.status != "ready"
    assert "method details" in state.remaining_issues
    assert "baseline" in state.remaining_issues
    assert "metrics" in state.remaining_issues
    assert "experimental setup" in state.remaining_issues
    question = controller.build_remaining_issue_question(state.remaining_issues)
    assert "核心机制" in question or "baseline" in question or "评价指标" in question


def test_process_user_clarification_fails_if_state_is_not_clarifying() -> None:
    controller = RequirementRefinementController(
        diagnoser=FakeDiagnoser([]),
        refiner=FakeRefiner("unused"),
    )
    state = RefinementState(
        original_instruction="清晰需求",
        current_instruction="清晰需求",
        status="ready",
    )

    with pytest.raises(ValueError):
        controller.process_user_clarification(state, "补充")


def test_first_stage_cli_parser_still_accepts_instruction() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["请诊断这个需求"])

    assert args.user_instruction == "请诊断这个需求"
    assert args.no_chain_of_thought is False


def test_interactive_cli_prints_remaining_issues_and_question(capsys: pytest.CaptureFixture[str]) -> None:
    class FakeController:
        @staticmethod
        def get_latest_diagnosis(state: RefinementState) -> InstructionDiagnosisResult:
            return state.diagnosis_history[-1]

        @staticmethod
        def build_remaining_issue_question(remaining_issues: str) -> str:
            return RequirementRefinementController.build_remaining_issue_question(
                remaining_issues
            )

    state = RefinementState(
        original_instruction="帮我设计实验",
        current_instruction="证明 FedRAA 在 CIFAR-10 上更好",
        remaining_issues="Missing method details, baseline, metrics, and experimental setup.",
        diagnosis_history=[
            make_diagnosis(
                "answer",
                explanation="需求表面上已补充方法和数据集。",
                clarification_question="",
            )
        ],
        status="clarifying",
    )

    _print_clarification_prompt(FakeController(), state)  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert "当前需求仍缺少以下信息" in output
    assert "method details" in output
    assert "澄清问题" in output
