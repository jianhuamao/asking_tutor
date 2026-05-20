# Instruction Diagnosis

Instruction Diagnosis is a first-stage prototype for a **Pre-answer Diagnostic Agent**. Before an LLM answers a user request, it performs a lightweight **Requirement Critique**: is the instruction clear, accurate, logically consistent, sufficiently constrained, and directly answerable?

This is not a prompt optimizer and not a full clarification agent. The first stage is intentionally single-turn: diagnose the instruction, then recommend one of three actions.

- `answer`: the instruction is good enough to answer now.
- `clarify`: one critical constraint is missing, so ask one question first.
- `revise`: the instruction has a contradiction, false premise, conceptual confusion, infeasible target, or oversized goal, so suggest a better formulation.

## What This Solves

Ordinary LLM direct answering often assumes the user instruction is valid and complete. A clarification agent focuses on asking follow-up questions to reduce ambiguity. This project sits before both: **Instruction Diagnosis** checks whether the instruction itself is clear, accurate, and logically sound before answering.

The long-term direction is **Socratic Requirement Refinement**, where the system can guide a user through multi-turn requirement repair. Stage 1 provides the single-turn diagnoser. Stage 2 adds a lightweight multi-turn refinement loop that repairs the requirement, but still does not answer the user's final task.

## Structure

```text
instruction_diagnosis/
  __init__.py
  signature.py
  diagnoser.py
  schema.py
  cli.py
  state.py
  refiner_signature.py
  refiner.py
  controller.py
  interactive_cli.py
tests/
  test_diagnoser.py
  test_refinement_state.py
  test_refiner.py
  test_controller.py
README.md
requirements.txt
```

## Install

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

Dependencies:

- `dspy-ai`
- `pydantic`
- `pytest`

## Configure DSPy

The code never hard-codes API keys. Configure DSPy in your application:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", api_key="YOUR_API_KEY")
dspy.configure(lm=lm)
```

For the CLI, set environment variables:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export DSPY_MODEL="openai/gpt-4o-mini"
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:DSPY_MODEL = "openai/gpt-4o-mini"
```

`DSPY_MODEL` is optional and defaults to `openai/gpt-4o-mini`.

## Run CLI

```bash
python -m instruction_diagnosis.cli "我想做一个完全无监督且绝对准确的LLM评测系统"
```

The CLI prints valid JSON.

## Stage 2: Socratic Requirement Refinement

Stage 1 is single-turn **Instruction Diagnosis**: it inspects one `user_instruction` and returns a structured diagnosis.

Stage 2 is multi-turn **Socratic Requirement Refinement**: it repeatedly diagnoses the current requirement, asks one focused clarification question or shows one revision suggestion, merges the user's clarification into a refined instruction, and diagnoses again.

Stage 2 does not directly answer the user's final question. Its output is a clearer, more accurate, more executable `refined_instruction` that can later be passed to a normal answering agent.

For experiment-design and method-comparison requests, Stage 2 treats missing method details, baselines, metrics, and experimental setup as blocking issues. User-defined method names such as `FedRAA`, `MyMethod`, `ProposedMethod`, or `Ours` are not assumed to be self-explanatory; the system should ask for the core mechanism or another key missing constraint instead of marking the requirement ready too early.

Run the interactive CLI:

```bash
python -m instruction_diagnosis.interactive_cli
```

Example dialogue:

```text
用户：
我想做一个完全无监督且绝对准确的LLM评测系统。

系统：
这个需求中“完全无监督”和“绝对准确”之间存在冲突。如果没有任何人工标注或外部参照，就很难定义并验证绝对准确。
你是否允许使用少量人工标注的验证集来校准自动评测器？

用户：
可以，但希望人工成本尽量低。

系统：
修正后的需求：
我想设计一个低人工成本的 LLM 自动评测系统，允许使用少量人工标注验证集进行校准，目标是在尽量减少人工标注成本的同时，提高自动评测结果与人工判断的一致性。

当前需求已经足够清晰，可以进入正式回答阶段。
```

The Stage 2 flow is implemented without LangGraph, web UI, external tools, web search, long-term memory, or model training. It is intentionally a small controller around `InstructionDiagnoser` and `InstructionRefiner`.

## Python Usage

```python
from instruction_diagnosis import InstructionDiagnoser

diagnoser = InstructionDiagnoser()
result = diagnoser.diagnose("请解释 FedAvg 和 FedProx 在 non-IID 联邦学习场景下的主要区别。")
print(result.action)
print(result.to_pretty_json())
```

`diagnose()` returns a Pydantic `InstructionDiagnosisResult`, not a raw string.

## Output Schema

```json
{
  "is_clear": true,
  "is_accurate": true,
  "is_logically_consistent": true,
  "has_missing_constraints": false,
  "has_ambiguity": false,
  "issue_type": "none",
  "explanation": "该问题目标明确，可以直接回答。",
  "revised_instruction": "",
  "clarification_question": "",
  "action": "answer"
}
```

Fields:

- `is_clear`: whether the instruction is expressed clearly.
- `is_accurate`: whether concepts and premises are accurate.
- `is_logically_consistent`: whether the instruction has no internal contradiction.
- `has_missing_constraints`: whether critical constraints are missing.
- `has_ambiguity`: whether obvious ambiguity exists.
- `issue_type`: one of `none`, `ambiguity`, `missing_constraint`, `logical_contradiction`, `conceptual_confusion`, `false_premise`, `too_broad`, `infeasible`, `underspecified`.
- `explanation`: friendly diagnosis.
- `revised_instruction`: clearer executable rewrite, or `""`.
- `clarification_question`: one most important question, or `""`.
- `action`: one of `answer`, `clarify`, `revise`.

## Examples

### 1. Clear Question

Input:

```text
请解释 FedAvg 和 FedProx 在 non-IID 联邦学习场景下的主要区别。
```

Expected:

```json
{
  "issue_type": "none",
  "action": "answer"
}
```

### 2. Missing Constraint

Input:

```text
帮我设计一个实验，证明我的方法比 baseline 好。
```

Expected behavior: `action = "clarify"` and `issue_type = "missing_constraint"` or `"underspecified"`. The clarification question should ask about the most critical missing element, such as task, dataset, baseline, or metric.

### 3. Logical Contradiction

Input:

```text
我想做一个完全无监督、完全不依赖人工标注、同时能保证评测结果绝对准确的 LLM 评测系统。
```

Example output:

```json
{
  "is_clear": false,
  "is_accurate": false,
  "is_logically_consistent": false,
  "has_missing_constraints": true,
  "has_ambiguity": false,
  "issue_type": "logical_contradiction",
  "explanation": "该需求同时要求完全无监督和绝对准确，但如果没有任何人工标注或外部参照，就很难定义和验证绝对准确。",
  "revised_instruction": "我想设计一个尽量减少人工标注的 LLM 自动评测系统，并使用少量 gold set 校准评测结果。",
  "clarification_question": "你是否允许使用少量人工标注的验证集来校准自动评测器？",
  "action": "revise"
}
```

### 4. Too Broad

Input:

```text
帮我写一篇能中 NeurIPS 的论文。
```

Expected behavior: `action = "clarify"` or `"revise"`, with `issue_type = "too_broad"`.

### 5. Conceptual Confusion

Input:

```text
GEPA 没有训练参数，为什么它的模型权重会随着 rollout 数量增加而更新？
```

Expected behavior: `action = "revise"` and `issue_type = "conceptual_confusion"`. The explanation should note that GEPA typically optimizes prompts, instructions, or programs, rather than directly updating model weights.

## Tests

The tests do not call a real LLM. They use fake predictor outputs to verify schema validation, enum restrictions, JSON parsing, diagnoser coercion, refiner fallback behavior, and controller state transitions.

```bash
pytest
```

## Extension Path

Stage 3 can extend this prototype with:

- LangGraph state machines.
- Web-based fact verification.
- External tool calls.
- Long-term memory.
- A Web UI.

To migrate toward a LangGraph-based **Socratic Requirement Refinement** flow later:

1. Keep `InstructionDiagnoser` as the first node.
2. Move the state transitions in `controller.py` into graph nodes and conditional edges.
3. Route `clarify` to a user-question node that asks only the single most important question.
4. Route `revise` to a requirement-rewrite node that proposes a repair and asks for confirmation.
5. Store user answers and revised instructions in graph state, then re-run diagnosis until the instruction is answerable or the loop limit is reached.

The current Stage 2 prototype deliberately avoids LangGraph and long-term memory so the diagnostic and refinement contracts stay small and testable.
