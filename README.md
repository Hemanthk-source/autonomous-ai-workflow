# Autonomous AI Workflow

An agent that takes a goal, plans it, executes each step with tools, reviews its own work, and produces a final report.

## Architecture

```
Goal -> Planner -> [ Executor (tool loop) -> Reviewer ] x N steps -> Final Report
                          ^________ retry with feedback ________|
```

## Setup

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"     # Windows: set ANTHROPIC_API_KEY=your-key
```

## Run

```bash
python autonomous_workflow.py "Create a 4-week Python study plan and save it as a file"
```

Outputs (deliverables, `final_report.md`, `run_log.json`) go to the `workspace/` folder.

## Extend it

- Add tools: write a function, register it in `TOOL_FUNCTIONS`, add its schema to `TOOL_SCHEMAS`.
- Add web search, email, database or API tools the same way.
- Change `MAX_RETRIES` / `MAX_TOOL_TURNS` to tune autonomy.
- Set `WORKFLOW_MODEL` to use a different Claude model.
