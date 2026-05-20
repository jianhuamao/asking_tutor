"""Command-line interface for Instruction Diagnosis."""

from __future__ import annotations

import argparse
import os
import sys

from .diagnoser import InstructionDiagnoser
from .config import configure_dspy

DEFAULT_MODEL = "openai/gpt-4o-mini"



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-turn Pre-answer Instruction Diagnosis."
    )
    parser.add_argument("user_instruction", help="The instruction to diagnose.")
    parser.add_argument(
        "--no-chain-of-thought",
        action="store_true",
        help="Use dspy.Predict instead of dspy.ChainOfThought.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        configure_dspy()
        diagnoser = InstructionDiagnoser(
            use_chain_of_thought=not args.no_chain_of_thought
        )
        result = diagnoser.diagnose(args.user_instruction)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(result.to_pretty_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
