## MANDATORY: Use td for Task Management

Run td usage --new-session at conversation start (or after /clear). This tells you what to work on next.

Sessions are automatic (based on terminal/agent context). Optional:
- td session "name" to label the current session
- td session --new to force a new session in the same context

Do NOT start a new session mid-work to satisfy td review rules. Use a real
reviewer sub-agent or separate agent context. An independent review is required;
the close may be delegated to any session.

You cannot review your own implementation, but you can close after an
independent review has been recorded. Under review_policy_mode=delegated:
  td approve <id> --record-only --reason "..."   # reviewer records approval
  td approve <id> --reason "using recorded approval"  # any session closes
