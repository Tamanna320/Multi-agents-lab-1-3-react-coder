"""
agent.py  ·  Lab 1.3 – ReACT Agent
────────────────────────────────────────────────────────────────────────────────
YOUR TASK
─────────
Implement the three functions marked with  TODO  below.  They wire together
the memory system (memory.py) and the tool layer (tools.py) into a working
ReACT loop.

The ReACT pattern structures LLM reasoning as a sequence of:
    Thought  → what the agent is planning to do next
    Action   → the tool to call (name + arguments)
    Observation → the tool's return value

The loop repeats until:
    • The LLM returns a  FINAL ANSWER:  marker (task complete), OR
    • The iteration counter reaches MAX_ITERATIONS (loop guard).

Files you will edit
───────────────────
    agent.py         ← THIS FILE.  Implement the three TODO functions.

Files provided (do not modify)
──────────────────────────────
    llm_client.py    – Helicone-routed OpenAI client + MODEL constant
    tools.py         – read_file, write_file, exec_python + dispatch_tool()
    memory.py        – SlidingWindowBuffer + SemanticMemory

Running the agent
─────────────────
    python agent.py

    The agent will attempt the default DEMO_TASK defined at the bottom of
    this file.  Swap it out with your own task to experiment.

Running the tests
─────────────────
    pytest tests/
"""

from __future__ import annotations

import json
from typing import Any
import re

from llm_client import client, MODEL
from memory import SemanticMemory, SlidingWindowBuffer
from tools import TOOL_SCHEMAS, dispatch_tool

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 10   # Hard ceiling on agentic loop iterations.


SYSTEM_PROMPT = """You are a Coder Agent. Your job is to complete coding tasks
by reasoning carefully and using the available tools.

## Available tools

You may use ONLY these three tools:
- read_file: Read a file from the workspace. Argument: {{"path": "filename"}}
- write_file: Write a file in the works
  {{"path": "filename", "content": "complete file content"}}
- exec_python: Execute Python code to test your work. Argument:
  {{"code": "Python source code"}}

Use the exact tool names: read_file, write_file, exec_python.
Never use namespaces, prefixes, parallel tool calls, or other tool names.
All file paths passed to read_file and write_file are relative to the workspace root.
NEVER include "workspace/" or "workspace\\" at the beginning of a file path.
For example, use "word_frequency.py", NOT "workspace/word_frequency.py".
The tools automatically add the workspace directory.
When executing Python, remember that code runs in a temporary script,
not necessarily from the workspace directory. Use workspace-relative paths
or absolute paths only when appropriate and permitted.

## Response format

Every response MUST use EXACTLY one of these formats.

FORMAT A - when calling a tool:
Thought: <brief explanation of the next step>
Action: <exact tool name>
Action Input: <single-line valid JSON object>

FORMAT B - when the task is complete:
Thought: <brief explanation of the result>
FINAL ANSWER: <deliverable, result, or relevant file path>

## Rules

- Always provide a Thought before every Action or FINAL ANSWER.
- Action Input must be a single-line valid JSON object.
- Make ONE tool call per response. Never bundle tools together.
- Never invent tool results; wait for the Observation.
- If a tool returns an ERROR, understand it and try a different approach.
- After writing code, verify it with exec_python whenever practical.
- exec_python runs a temporary Python script. Do not assume its current
  working directory is the project root or workspace.
- Do not assume workspace modules can be imported by their filename alone.
- For testing a generated file, use read_file to inspect its contents first.
  Then use exec_python with a self-contained test when practical, or explicitly
  set up the correct import path and file paths.
- Pass workspace-relative paths to read_file and write_file, without a
  workspace/ prefix.
- For exec_python, include any required setup and file paths in the code.
- Stop and return FINAL ANSWER once the task is complete.
- You have a maximum of {max_iter} iterations. Use them wisely.
""".format(max_iter=MAX_ITERATIONS)



# ── Output schema ─────────────────────────────────────────────────────────────

class AgentResult:
    """Structured result returned by run_agent()."""

    def __init__(
        self,
        answer: str,
        iterations: int,
        tool_calls: list[dict[str, Any]],
        stopped_early: bool,
    ) -> None:
        self.answer = answer          # The FINAL ANSWER text.
        self.iterations = iterations  # How many loop iterations ran.
        self.tool_calls = tool_calls  # List of {"tool": ..., "args": ..., "result": ...}.
        self.stopped_early = stopped_early  # True if MAX_ITERATIONS was reached.

    def __repr__(self) -> str:
        status = "MAX_ITER" if self.stopped_early else "DONE"
        return (
            f"AgentResult(status={status}, iterations={self.iterations}, "
            f"tool_calls={len(self.tool_calls)})\n"
            f"ANSWER:\n{self.answer}"
        )


# ── Functions for students to implement ───────────────────────────────────────

def build_system_prompt(semantic_memory: SemanticMemory, task: str) -> str:
    """
    Construct the system prompt by injecting relevant long-term memories.

    Steps
    -----
    1. Call  semantic_memory.retrieve(task, top_k=3)  to get up to three
       relevant memory strings.
    2. If any memories were returned, format them as a bulleted list and
       append them to SYSTEM_PROMPT under a  ## Relevant memories  heading.
    3. Return the final system prompt string.

    If the memory store is empty (semantic_memory.count() == 0), return
    SYSTEM_PROMPT unchanged.

    Parameters
    ----------
    semantic_memory : SemanticMemory
        The long-term vector store.  May be empty on the first run.
    task : str
        The user's task description.  Used as the retrieval query.

    Returns
    -------
    str
        The complete system prompt, with memories appended if available.
    """
    if semantic_memory.count() == 0:
        return SYSTEM_PROMPT

    memories = semantic_memory.retrieve(task, top_k=3)

    if not memories:
        return SYSTEM_PROMPT

    memory_section = "\n\n## Relevant memories\n"
    memory_section += "\n".join(f"- {memory}" for memory in memories)

    return SYSTEM_PROMPT + memory_section


def parse_llm_response(response_text: str) -> dict[str, Any]:
    """
    Parse a single LLM response into a structured dict.

    The response will be in one of two formats (see SYSTEM_PROMPT):

    FORMAT A (tool call):
        Thought: <text>
        Action: <tool_name>
        Action Input: <json>

    FORMAT B (final answer):
        Thought: <text>
        FINAL ANSWER: <text>

    Steps
    -----
    1. Extract the  Thought:  line.
    2. Check whether  "FINAL ANSWER:"  appears in the response.
       • If yes → return {"thought": ..., "final_answer": <everything after "FINAL ANSWER:">}
    3. Otherwise, extract  Action:  and  Action Input: .
       • Parse Action Input as JSON with json.loads().
       • If json.loads() raises, return an error dict so the loop can
         feed an error observation back to the LLM.
    4. Return {"thought": ..., "action": ..., "action_input": <dict>}

    Error return format (malformed Action Input):
        {"thought": ..., "action": ..., "parse_error": <error message string>}

    Hints
    -----
    • Use  str.split("Thought:", 1)  /  str.split("Action:", 1)  etc. to
      extract each field.
    • Strip whitespace from every extracted value.
    • The Action Input JSON may contain newlines — extract everything after
      "Action Input:" up to the end of the string (or next recognised keyword).

    Parameters
    ----------
    response_text : str
        Raw text from the LLM's message content.

    Returns
    -------
    dict
        A structured dict with the keys described above.
    """

    thought = ""
    thought_match = re.search(
        r"Thought:\s*(.*?)(?=\n(?:Action:|FINAL ANSWER:)|\Z)",
        response_text,
        re.DOTALL,
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    final_match = re.search(r"FINAL ANSWER:\s*(.*)", response_text, re.DOTALL)
    if final_match:
        return {
            "thought": thought,
            "final_answer": final_match.group(1).strip(),
        }

    action_match = re.search(r"Action:\s*(.+)", response_text)
    input_match = re.search(r"Action Input:\s*(.*)", response_text, re.DOTALL)

    action = action_match.group(1).strip() if action_match else ""

    if not action or not input_match:
        return {
            "thought": thought,
            "parse_error": "Could not find a valid Action and Action Input.",
        }

    try:
        action_input = json.loads(input_match.group(1).strip())
    except json.JSONDecodeError as error:
        return {
            "thought": thought,
            "action": action,
            "parse_error": str(error),
        }

    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
    }



def run_agent(task: str) -> AgentResult:
    """
    Run the full ReACT agentic loop for the given *task*.

    Algorithm
    ---------
    1. Initialise a SlidingWindowBuffer and a SemanticMemory.
    2. Call build_system_prompt() and pass the result to buffer.set_system().
    3. Add the user task as the first message:  buffer.add("user", task)
    4. Enter the loop (up to MAX_ITERATIONS):
        a. Call the LLM with  client.chat.completions.create()  passing
           buffer.messages() and TOOL_SCHEMAS as the tools list.
           Use  tool_choice="none"  so the model uses the ReACT text format
           rather than native function-calling.
        b. Extract the assistant's text: response.choices[0].message.content
        c. Add the assistant message to the buffer:
               buffer.add("assistant", <assistant_text>)
        d. Parse the response with  parse_llm_response().
        e. If the parsed result contains "final_answer":
               • Store the final answer in semantic memory:
                     semantic_memory.store("Task: " + task + "\\nAnswer: " + final_answer)
               • Return an AgentResult with stopped_early=False.
        f. If the parsed result contains "parse_error":
               • Build an observation string:
                     "OBSERVATION: Parse error – " + parse_error
               • Add it to the buffer as a user message and continue.
        g. Otherwise (normal tool call):
               • Call  dispatch_tool(action, action_input)  to get the result.
               • Record the call in the tool_calls list.
               • Build the observation:
                     "OBSERVATION: " + tool_result
               • Store noteworthy observations in semantic memory
                 (store if the observation does NOT start with "ERROR").
               • Add the observation to the buffer as a user message.
    5. If the loop ends without a FINAL ANSWER, return AgentResult with
       stopped_early=True and answer="MAX_ITERATIONS reached".

    Parameters
    ----------
    task : str
        Natural language description of the coding task to complete.

    Returns
    -------
    AgentResult
        Structured result including the final answer, iteration count,
        and a log of all tool calls made.
    """
    
    buffer = SlidingWindowBuffer()
    semantic_memory = SemanticMemory()
    tool_calls = []

    system_prompt = build_system_prompt(semantic_memory, task)
    buffer.set_system(system_prompt)
    buffer.add("user", task)

    for iteration in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=buffer.messages(),
            tools=TOOL_SCHEMAS,
            tool_choice="none",
        )

        assistant_text = response.choices[0].message.content or ""
        buffer.add("assistant", assistant_text)

        parsed = parse_llm_response(assistant_text)

        if "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            semantic_memory.store(
                "Task: " + task + "\nAnswer: " + final_answer
            )

            return AgentResult(
                answer=final_answer,
                iterations=iteration,
                tool_calls=tool_calls,
                stopped_early=False,
            )

        if "parse_error" in parsed:
            observation = "OBSERVATION: Parse error – " + parsed["parse_error"]
            buffer.add("user", observation)
            continue

        action = parsed["action"]
        action_input = parsed["action_input"]

        tool_result = dispatch_tool(action, action_input)

        tool_calls.append({
            "tool": action,
            "args": action_input,
            "result": tool_result,
        })

        observation = "OBSERVATION: " + str(tool_result)

        if not observation.startswith("OBSERVATION: ERROR"):
            semantic_memory.store(observation)

        buffer.add("user", observation)

    return AgentResult(
        answer="MAX_ITERATIONS reached",
        iterations=MAX_ITERATIONS,
        tool_calls=tool_calls,
        stopped_early=True,
    )



# ── Entry point ───────────────────────────────────────────────────────────────

DEMO_TASK = (
    "Write a Python function called word_frequency(filepath: str) -> dict "
    "that reads a text file and returns a dictionary mapping each word to "
    "the number of times it appears.  The function should be case-insensitive "
    "and ignore punctuation. Write the function to word_frequency.py, "
    "then write a small test script to test_word_frequency.py that "
    "creates a sample file, calls the function, and prints the result.  "
    "Finally, run the test script and confirm it produces output."
)

if __name__ == "__main__":
    result = run_agent(DEMO_TASK)
    print(result)

    for index, call in enumerate(result.tool_calls, start=1):
        print(f"\n--- Tool Call {index} ---")
        print("Tool:", call["tool"])
        print("Arguments:", call["args"])
        print("Result:", call["result"])
