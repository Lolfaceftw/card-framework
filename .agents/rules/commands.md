---
trigger: always_on
---

<package_manager>
uv
</package_manager>
<terminal_rules>
- Use semicolon for command separation.
</terminal_rules>
<additional_commands>
- `git`
- `gh`
</additional_commands>
<gemini_cli_usage>
  <trigger>
    - User says "using gemini" or similar
  </trigger>
  <tool>
- Use terminal via `gemini --prompt "<query>"`
  </tool>
  <query_rules>
    <inject_before_query>
  You will strictly only plan, no code changes. You will make use of Web Search for all queries to ensure latest information. You must check the current date. Query: 
    </inject_before_query>
  </query_rules>

</gemini_cli_usage>