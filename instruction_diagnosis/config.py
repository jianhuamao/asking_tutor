import os
import dspy


def configure_dspy():
    model = os.getenv("DSPY_MODEL", "openai/deepseek-chat")
    api_key = os.getenv("DSPY_API_KEY")
    api_base = os.getenv("DSPY_API_BASE", "https://api.deepseek.com/anthropic")

    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please set environment variable DEEPSEEK_API_KEY."
        )

    lm = dspy.LM(
        model,
        api_key=api_key,
        api_base=api_base,
    )

    dspy.configure(lm=lm)
