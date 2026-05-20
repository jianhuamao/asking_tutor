"""Controller for Socratic Requirement Refinement."""

from __future__ import annotations

from typing import Any, Protocol

from .diagnoser import InstructionDiagnoser
from .refiner import InstructionRefiner
from .state import InstructionRefinementResult
from .state import RefinementState


class DiagnoserProtocol(Protocol):
    def diagnose(self, user_instruction: str) -> Any:
        ...


class RefinerProtocol(Protocol):
    def refine(
        self,
        original_instruction: str,
        current_instruction: str,
        diagnosis_explanation: str,
        user_clarification: str,
    ) -> InstructionRefinementResult:
        ...


class RequirementRefinementController:
    """Coordinate diagnosis, clarification, and refinement state transitions."""

    CRITICAL_REMAINING_ISSUE_KEYWORDS = (
        "method detail",
        "method details",
        "method mechanism",
        "core mechanism",
        "method summary",
        "baseline",
        "baselines",
        "metric",
        "metrics",
        "evaluation metric",
        "experimental setup",
        "experiment setup",
        "federated setting",
        "federated learning setting",
        "non-iid",
        "noise setting",
        "方法细节",
        "核心机制",
        "方法摘要",
        "基线",
        "对比方法",
        "评价指标",
        "评估指标",
        "实验设置",
        "联邦学习设置",
        "数据异质性",
        "噪声设置",
    )

    NON_ISSUE_PHRASES = (
        "none",
        "n/a",
        "no remaining issues",
        "no explicit remaining issues",
        "没有",
        "无",
        "无剩余问题",
    )

    def __init__(
        self,
        diagnoser: DiagnoserProtocol | None = None,
        refiner: RefinerProtocol | None = None,
    ) -> None:
        self.diagnoser = diagnoser or InstructionDiagnoser()
        self.refiner = refiner or InstructionRefiner()

    def start(self, user_instruction: str, max_rounds: int = 3) -> RefinementState:
        """Start a new refinement session with one diagnosis pass."""

        state = RefinementState(
            original_instruction=user_instruction,
            current_instruction=user_instruction,
            max_rounds=max_rounds,
            status="diagnosing",
        )

        diagnosis = self.diagnoser.diagnose(user_instruction)
        state.diagnosis_history.append(diagnosis)

        if diagnosis.action == "answer":
            state.status = "ready"
            state.refined_instruction = state.current_instruction
        else:
            state.status = "clarifying"

        return state

    def process_user_clarification(
        self,
        state: RefinementState,
        user_clarification: str,
    ) -> RefinementState:
        """Refine with one user clarification and diagnose the new instruction."""

        if state.status != "clarifying":
            raise ValueError("process_user_clarification requires clarifying status.")

        state.user_clarifications.append(user_clarification)
        state.round_id += 1

        if state.round_id > state.max_rounds:
            state.status = "failed"
            return state

        state.status = "refining"
        latest_diagnosis = self.get_latest_diagnosis(state)
        refinement = self.refiner.refine(
            original_instruction=state.original_instruction,
            current_instruction=state.current_instruction,
            diagnosis_explanation=getattr(latest_diagnosis, "explanation", ""),
            user_clarification=user_clarification,
        )

        state.refined_instruction = refinement.refined_instruction
        state.current_instruction = refinement.refined_instruction
        state.remaining_issues = refinement.remaining_issues.strip()

        state.status = "diagnosing"
        new_diagnosis = self.diagnoser.diagnose(state.current_instruction)
        state.diagnosis_history.append(new_diagnosis)

        has_blocking_issues = self.has_critical_remaining_issues(state.remaining_issues)

        if new_diagnosis.action == "answer" and not has_blocking_issues:
            state.status = "ready"
        elif state.round_id >= state.max_rounds:
            state.status = "failed"
        else:
            state.status = "clarifying"

        return state

    @staticmethod
    def get_latest_diagnosis(state: RefinementState) -> Any:
        """Return the latest diagnosis in a state."""

        if not state.diagnosis_history:
            raise ValueError("No diagnosis is available in the refinement state.")
        return state.diagnosis_history[-1]

    @classmethod
    def has_critical_remaining_issues(cls, remaining_issues: str) -> bool:
        """Return whether remaining issues should block the ready state."""

        text = remaining_issues.strip().lower()
        if not text:
            return False
        if text in cls.NON_ISSUE_PHRASES:
            return False
        return any(keyword in text for keyword in cls.CRITICAL_REMAINING_ISSUE_KEYWORDS)

    @classmethod
    def build_remaining_issue_question(cls, remaining_issues: str) -> str:
        """Ask one focused question based on blocking remaining issues."""

        text = remaining_issues.lower()
        if (
            "method detail" in text
            or "core mechanism" in text
            or "方法细节" in text
            or "核心机制" in text
        ):
            return "请用一两句话说明你的方法的核心机制是什么？"
        if "baseline" in text or "基线" in text or "对比方法" in text:
            return "你希望主要和哪些 baseline 或对比方法进行比较？"
        if "metric" in text or "评价指标" in text or "评估指标" in text:
            return "你希望用哪个最关键的评价指标来证明方法更好？"
        if "setup" in text or "设置" in text:
            return "请补充一个最关键的实验设置，例如 non-IID 划分、联邦学习配置或噪声设置。"
        return "请补充当前需求中最关键的一项缺失信息。"
