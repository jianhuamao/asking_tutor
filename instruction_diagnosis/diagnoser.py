"""Instruction Diagnosis module powered by DSPy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .schema import InstructionDiagnosisResult


Predictor = Callable[..., Any]


class InstructionDiagnoser:
    """Single-turn Pre-answer Instruction Diagnosis.

    The default predictor is a DSPy ChainOfThought over
    InstructionDiagnosisSignature. Tests and downstream systems can inject a fake
    predictor with the same call shape to avoid real LLM calls.
    """

    def __init__(
        self,
        predictor: Predictor | None = None,
        *,
        use_chain_of_thought: bool = True,
    ) -> None:
        self.predictor = predictor or self._build_default_predictor(
            use_chain_of_thought=use_chain_of_thought
        )

    def diagnose(self, user_instruction: str) -> InstructionDiagnosisResult:
        """Diagnose one user instruction and return a Pydantic object."""

        if not isinstance(user_instruction, str):
            raise TypeError("user_instruction must be a string.")

        if not user_instruction.strip():
            return InstructionDiagnosisResult(
                is_clear=False,
                is_accurate=True,
                is_logically_consistent=True,
                has_missing_constraints=True,
                has_ambiguity=False,
                issue_type="underspecified",
                explanation="用户输入为空，缺少需要诊断或回答的具体 instruction。",
                revised_instruction="",
                clarification_question="你希望系统诊断或回答的具体问题是什么？",
                action="clarify",
            )

        prediction = self.predictor(user_instruction=user_instruction)
        return self._coerce_prediction(prediction)

    @staticmethod
    def _build_default_predictor(*, use_chain_of_thought: bool) -> Predictor:
        try:
            import dspy
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "DSPy is not installed. Install dependencies with "
                "`pip install -r requirements.txt` before using the real LLM "
                "diagnoser."
            ) from exc

        from .signature import InstructionDiagnosisSignature

        predictor_cls = dspy.ChainOfThought if use_chain_of_thought else dspy.Predict
        return predictor_cls(InstructionDiagnosisSignature)

    @staticmethod
    def _coerce_prediction(prediction: Any) -> InstructionDiagnosisResult:
        """Convert DSPy/fake predictor output into the public schema."""

        if isinstance(prediction, InstructionDiagnosisResult):
            return prediction

        if isinstance(prediction, str):
            return InstructionDiagnosisResult.from_json(prediction)

        if isinstance(prediction, dict):
            if "diagnosis_json" in prediction:
                return InstructionDiagnoser._coerce_prediction(
                    prediction["diagnosis_json"]
                )
            return InstructionDiagnosisResult.from_mapping(prediction)

        diagnosis_json = getattr(prediction, "diagnosis_json", None)
        if diagnosis_json is not None:
            return InstructionDiagnoser._coerce_prediction(diagnosis_json)

        model_fields = getattr(InstructionDiagnosisResult, "model_fields", None)
        field_names = (
            model_fields.keys()
            if model_fields is not None
            else InstructionDiagnosisResult.__fields__.keys()
        )
        data = {
            field: getattr(prediction, field)
            for field in field_names
            if hasattr(prediction, field)
        }
        if data:
            return InstructionDiagnosisResult.from_mapping(data)

        raise TypeError(
            "Predictor output must be a JSON string, dict, Pydantic result, "
            "or object with a diagnosis_json attribute."
        )
