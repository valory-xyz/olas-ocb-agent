"""Connect's read-only tools, until Connect flags them itself.

Interim: goes away once Connect marks its read-only tools with MCP's `readOnlyHint`;
the runtime already prefers Connect's own flags as soon as its tool list carries any (`policy.read_only_from`).
"""

FALLBACK_READ_ONLY = {"wallet_info", "mech_tools", "mech_result", "transaction_status", "preflight_transaction"}
