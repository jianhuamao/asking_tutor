"""Interactive CLI for Socratic Requirement Refinement."""

from __future__ import annotations

import sys

from .config import configure_dspy
from .controller import RequirementRefinementController
from .state import RefinementState


EXIT_COMMANDS = {"exit", "quit", "q"}


def _print_ready(state: RefinementState) -> None:
    final_instruction = state.refined_instruction or state.current_instruction
    print("\n当前需求已经足够清晰，可以进入正式回答阶段。")
    print("最终需求：")
    print(final_instruction)


def _print_failed(state: RefinementState) -> None:
    best_instruction = state.refined_instruction or state.current_instruction
    print("\n已达到最大澄清轮数，当前需求仍可能存在问题。")
    print("当前最优修正版需求：")
    print(best_instruction)


def _print_clarification_prompt(
    controller: RequirementRefinementController,
    state: RefinementState,
) -> None:
    diagnosis = controller.get_latest_diagnosis(state)
    explanation = getattr(diagnosis, "explanation", "")
    clarification_question = getattr(diagnosis, "clarification_question", "")
    revised_instruction = getattr(diagnosis, "revised_instruction", "")
    remaining_issues = state.remaining_issues.strip()

    print("\n诊断结果：")
    print(explanation or "当前需求仍需要进一步澄清。")

    if remaining_issues:
        print("\n当前需求仍缺少以下信息：")
        print(remaining_issues)

    if clarification_question:
        print("\n澄清问题：")
        print(clarification_question)
    elif remaining_issues:
        print("\n澄清问题：")
        print(controller.build_remaining_issue_question(remaining_issues))
    elif revised_instruction:
        print("\n修改建议：")
        print(revised_instruction)


def main() -> int:
    try:
        configure_dspy()
    except Exception as exc:
        print(f"模型配置失败：{exc}", file=sys.stderr)
        return 1

    try:
        user_instruction = input("请输入你的需求：").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。")
        return 0

    if not user_instruction:
        print("需求不能为空。", file=sys.stderr)
        return 1
    if user_instruction.lower() in EXIT_COMMANDS:
        print("已退出。")
        return 0

    controller = RequirementRefinementController()

    try:
        state = controller.start(user_instruction)
    except Exception as exc:
        print(f"诊断失败：{exc}", file=sys.stderr)
        return 1

    while True:
        if state.status == "ready":
            _print_ready(state)
            return 0
        if state.status == "failed":
            _print_failed(state)
            return 0
        if state.status != "clarifying":
            print(f"流程状态异常：{state.status}", file=sys.stderr)
            return 1

        _print_clarification_prompt(controller, state)

        try:
            user_input = input("\n请输入你的补充说明：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if user_input.lower() in EXIT_COMMANDS:
            print("已退出。")
            return 0
        if not user_input:
            print("补充说明不能为空。")
            continue

        try:
            state = controller.process_user_clarification(state, user_input)
        except Exception as exc:
            print(f"需求修正失败：{exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
