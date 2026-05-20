"""DSPy signature for requirement refinement."""

from __future__ import annotations

try:
    import dspy
except ImportError as exc:  # pragma: no cover - exercised only without dependency
    raise ImportError(
        "DSPy is required for InstructionRefinementSignature. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


class InstructionRefinementSignature(dspy.Signature):
    """Refine a user requirement without answering the final task.

    You are a requirement refiner, not a final-task answerer.
    Your job is to combine the original user requirement and the user's latest
    clarification into a clearer, more accurate, more executable instruction.

    Rules:
    1. refined_instruction must preserve the user's original intent.
    2. refined_instruction must absorb the user's clarification.
    3. refined_instruction should reduce ambiguity, missing constraints, logical
       contradictions, and conceptual confusion noted in diagnosis_explanation.
    4. Do not invent constraints or facts the user did not provide.
    5. If information is still missing, explain it in remaining_issues.
    6. refined_instruction should be a clear question or task that can be handed
       to an ordinary LLM for answering.
    7. Do not put follow-up questions such as "please clarify whether..." inside
       refined_instruction.
    8. Do not answer refined_instruction. Only output the repaired requirement.
    9. Do not hide missing details behind vague phrases such as "relevant
       baselines", "appropriate metrics", or "clear setup". If baseline,
       metrics, method details, or experimental setup are not provided, write
       those missing items explicitly in remaining_issues.
    10. refined_instruction may organize the information already supplied, but it
        must not pretend the requirement is complete when critical experimental
        details are still missing.
    """

    original_instruction: str = dspy.InputField(
        desc="The user's first raw requirement."
    )
    current_instruction: str = dspy.InputField(
        desc="The requirement currently being diagnosed."
    )
    diagnosis_explanation: str = dspy.InputField(
        desc="The latest diagnosis explanation that identified the main issue."
    )
    user_clarification: str = dspy.InputField(
        desc="The user's answer or correction after the diagnosis."
    )
    refined_instruction: str = dspy.OutputField(
        desc="A clearer, more accurate, more executable instruction."
    )
    remaining_issues: str = dspy.OutputField(
        desc="Remaining missing information or issues, or an empty string."
    )
