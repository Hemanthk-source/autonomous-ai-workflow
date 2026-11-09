"""
Autonomous AI Workflow
----------------------
Give it a goal. It will:
  1. PLAN    - break the goal into small steps
  2. EXECUTE - carry out each step using tools (agent loop)
  3. REVIEW  - check each result, retry if it is not good enough
  4. REPORT  - write a final summary and save the full run log

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python autonomous_workflow.py "Create a study plan for learning Python in 4 weeks"
"""

import ast
import json
import operator
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# ---------------------------------------------------------------- config ----
MODEL = os.getenv("WORKFLOW_MODEL", "claude-sonnet-5-5")
WORKSPACE = Path(os.getenv("WORKFLOW_WORKSPACE", "workspace")).resolve()
MAX_RETRIES = 2          # retries per step if the reviewer rejects the result
MAX_TOOL_TURNS = 8       # safety cap on tool calls per attempt
WORKSPACE.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


# ----------------------------------------------------------------- tools ----
def _safe_path(name: str) -> Path:
    """Keep all file access inside the workspace folder."""
    path = (WORKSPACE / name).resolve()
    if WORKSPACE not in path.parents and path != WORKSPACE:
        raise ValueError("Access outside the workspace is not allowed.")
    return path


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    return str(_eval(ast.parse(expression, mode="eval").body))


def write_file(filename: str, content: str) -> str:
    path = _safe_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Saved {len(content)} characters to {filename}"


def read_file(filename: str) -> str:
    return _safe_path(filename).read_text(encoding="utf-8")


def list_files() -> str:
    files = [str(p.relative_to(WORKSPACE)) for p in WORKSPACE.rglob("*") if p.is_file()]
    return "\n".join(files) or "(workspace is empty)"


TOOL_FUNCTIONS = {
    "calculator": calculator,
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
}

TOOL_SCHEMAS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression, e.g. '(12*4)/3'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "write_file",
        "description": "Save text content to a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "list_files",
        "description": "List all files currently in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ------------------------------------------------------------ LLM helpers ----
def ask(system: str, user: str, max_tokens: int = 1500) -> str:
    """Single LLM call, no tools."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def ask_json(system: str, user: str):
    """LLM call that must return JSON."""
    text = ask(system + "\nRespond with valid JSON only. No markdown fences.", user)
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# ------------------------------------------------------------ 1. planner ----
def plan(goal: str) -> list[dict]:
    steps = ask_json(
        "You are a planning agent. Break the goal into 3-6 small, concrete, "
        "ordered steps that an agent with file and calculator tools can do. "
        'Return a JSON list: [{"id": 1, "task": "...", "success_criteria": "..."}]',
        f"Goal: {goal}",
    )
    return steps


# ---------------------------------------------------------- 2. executor ----
def execute_step(goal: str, step: dict, memory: list[str], feedback: str = "") -> str:
    """Agent loop: the model may call tools repeatedly until it finishes."""
    system = (
        "You are an autonomous worker agent. Complete the current step using "
        "the tools when helpful. Save any deliverables with write_file. "
        "When done, reply with a short summary of what you did and the result."
    )
    context = "\n".join(memory) or "(nothing yet)"
    prompt = (
        f"Overall goal: {goal}\n\n"
        f"Completed so far:\n{context}\n\n"
        f"Current step: {step['task']}\n"
        f"Success criteria: {step['success_criteria']}\n"
    )
    if feedback:
        prompt += f"\nPrevious attempt was rejected. Reviewer feedback: {feedback}\n"

    messages = [{"role": "user", "content": prompt}]

    for _ in range(MAX_TOOL_TURNS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip()

        results = []
        for block in resp.content:
            if block.type == "tool_use":
                print(f"      tool -> {block.name}({json.dumps(block.input)[:80]})")
                try:
                    output = TOOL_FUNCTIONS[block.name](**block.input)
                except Exception as e:  # report errors back so the agent can recover
                    output = f"Error: {e}"
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": str(output)}
                )
        messages.append({"role": "user", "content": results})

    return "Stopped: reached the maximum number of tool turns."


# ----------------------------------------------------------- 3. reviewer ----
def review(step: dict, result: str) -> dict:
    return ask_json(
        "You are a strict reviewer. Decide if the result meets the success "
        'criteria. Return JSON: {"approved": true/false, "feedback": "..."}',
        f"Step: {step['task']}\nCriteria: {step['success_criteria']}\n"
        f"Files in workspace:\n{list_files()}\n\nResult:\n{result}",
    )


# ------------------------------------------------------------ orchestrator ----
def run(goal: str) -> None:
    log = {"goal": goal, "started": datetime.now().isoformat(), "steps": []}
    print(f"\nGOAL: {goal}\n")

    print("Planning...")
    steps = plan(goal)
    for s in steps:
        print(f"  {s['id']}. {s['task']}")

    memory: list[str] = []

    for step in steps:
        print(f"\nStep {step['id']}: {step['task']}")
        feedback, result, verdict = "", "", {"approved": False, "feedback": ""}

        for attempt in range(1, MAX_RETRIES + 2):
            print(f"   attempt {attempt}")
            result = execute_step(goal, step, memory, feedback)
            verdict = review(step, result)
            print(f"   review: {'approved' if verdict['approved'] else 'rejected'}")
            if verdict["approved"]:
                break
            feedback = verdict["feedback"]

        memory.append(f"Step {step['id']} ({step['task']}): {result}")
        log["steps"].append(
            {"step": step, "result": result, "approved": verdict["approved"],
             "feedback": verdict["feedback"]}
        )

    print("\nWriting final report...")
    report = ask(
        "You write clear final reports for completed workflows.",
        f"Goal: {goal}\n\nWork completed:\n" + "\n".join(memory)
        + f"\n\nFiles created:\n{list_files()}\n\nWrite a concise final report.",
        max_tokens=1200,
    )
    log["report"] = report
    log["finished"] = datetime.now().isoformat()

    (WORKSPACE / "final_report.md").write_text(report, encoding="utf-8")
    (WORKSPACE / "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\nOutputs saved in: {WORKSPACE}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python autonomous_workflow.py "your goal here"')
        sys.exit(1)
    run(" ".join(sys.argv[1:]))
