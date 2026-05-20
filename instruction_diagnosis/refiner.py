"""Requirement refiner powered by DSPy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .state import InstructionRefinementResult


Predictor = Callable[..., Any]


class InstructionRefiner:
    """Merge user clarification into a refined instruction."""

    def __init__(
        self,
        predictor: Predictor | None = None,
        *,
        use_chain_of_thought: bool = True,
    ) -> None:
        self.predictor = predictor or self._build_default_predictor(
            use_chain_of_thought=use_chain_of_thought
        )

    def refine(
        self,
        original_instruction: str,
        current_instruction: str,
        diagnosis_explanation: str,
        user_clarification: str,
    ) -> InstructionRefinementResult:
        """Return a structured refined instruction."""

        prediction = self.predictor(
            original_instruction=original_instruction,
            current_instruction=current_instruction,
            diagnosis_explanation=diagnosis_explanation,
            user_clarification=user_clarification,
        )
        return self._coerce_prediction(
            prediction,
            current_instruction=current_instruction,
            user_clarification=user_clarification,
        )

    @staticmethod
    def _build_default_predictor(*, use_chain_of_thought: bool) -> Predictor:
        try:
            import dspy
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "DSPy is not installed. Install dependencies with "
                "`pip install -r requirements.txt` before using the refiner."
            ) from exc

        from .refiner_signature import InstructionRefinementSignature

        predictor_cls = dspy.ChainOfThought if use_chain_of_thought else dspy.Predict
        return predictor_cls(InstructionRefinementSignature)

    @staticmethod
    def _coerce_prediction(
        prediction: Any,
        *,
        current_instruction: str,
        user_clarification: str,
    ) -> InstructionRefinementResult:
        if isinstance(prediction, InstructionRefinementResult):
            result = prediction
        elif isinstance(prediction, dict):
            result = InstructionRefinementResult.from_mapping(prediction)
        else:
            data = {}
            for field in ("refined_instruction", "remaining_issues"):
                if hasattr(prediction, field):
                    data[field] = getattr(prediction, field)
            if data:
                result = InstructionRefinementResult.from_mapping(data)
            else:
                result = InstructionRefiner._fallback_result(
                    current_instruction=current_instruction,
                    user_clarification=user_clarification,
                )

        if result.refined_instruction.strip():
            return result

        return InstructionRefiner._fallback_result(
            current_instruction=current_instruction,
            user_clarification=user_clarification,
            remaining_issues=result.remaining_issues,
        )

    @staticmethod
    def _fallback_result(
        *,
        current_instruction: str,
        user_clarification: str,
        remaining_issues: str = "",
    ) -> InstructionRefinementResult:
        clarification = user_clarification.strip()
        if clarification:
            refined_instruction = (
                f"{current_instruction.strip()}\n\n补充说明：{clarification}"
            )
        else:
            refined_instruction = current_instruction.strip()

        return InstructionRefinementResult(
            refined_instruction=refined_instruction,
            remaining_issues=(
                remaining_issues.strip()
                or "The model did not provide explicit remaining issues."
            ),
        )
