# Scope
These instructions apply to all AI agents and contributors in this repository. Follow them for all Python code and prompt design work.

When the contract says "refer to @<file_md>", attach the prefix to be "@agentic_coder_prompts/skills/" so it is now "refer to "@<prefix>/<file_md>" 

# Code Writing Contract

When modifying/changing parts of the code, refer to:

References:
1. IF it involves a function
1.1 You are to write a docstring, refer to @python_docstrings.md
1.2 You are to implement typehinting, refer to @python_typehinting.md
2. ON Logging statements, refer to @python_logging.md   
2.1 Implement if none 
2.2 If there is a code that is to be changed that logging is affected, change logging accordingly where approriate.  
3. ON design and architecture, refer to @code_structure_and_architecture.md.
4. ON design patterns, refer to @python_design_patterns.md
5. ON quality checks, @prompt_engineering.md
6. ON type hinting, refer to @python_typehinting.md
7. ON codebase context, refer to @codebase.md

Flows:
1. Ensure that proper logging is implemented.
2. Ensure that proper design patterns and architecture are implemented.
3. Ensure quality checks.
4. Ensure type checks pass using mypy or pyright.
5. Ensure lint and format pass using ruff + formatter.
6. Ensure all tests pass.
7. Ensure regression tests pass.
8. Spawn one adversarial subagent to criticize the changes made on the code. Prompt must include the phrase "You are a merciless rational critic. State at the end 'SATISFIED' if you are satisfied."
9. Synthesize the adversarial subagent findings, and repeat 1-7.

## Contract End Agreement
If and only if the adversarial subagent is SATISFIED.

# Plan Mode Contract
Flows:
1. Spawn different subagents for the query to research and plan.
1.1 FOR any requests that involve querying about AI and/or optimization, prioritize academic journals and their benchmark numbers. State the numbers from the benchmark, and why the benchmark is chosen.
2. Synthesize the results.
3. Spawn one adversarial subagent to criticize the synthesis. Prompt must include the phrase "You are a merciless rational critic."
4. Synthesize the result.
4.1 You are allowed to use Web Search to refine the plan.
## Contract End Agreement
If you are done with Step 4.

