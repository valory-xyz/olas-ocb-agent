# Security policy

olas-ocb-agent runs an AI agent next to a wallet. If you find a way to get around its protections, please tell us
privately first.

## Reporting a vulnerability

Report it through GitHub's **private vulnerability reporting** on this repository: the *Security* tab, then
*Report a vulnerability* ([direct link](https://github.com/valory-xyz/olas-ocb-agent/security/advisories/new)).
Don't open a public issue or pull request, and don't post details elsewhere until a fix is released.

Please include the version (`ocb-agent --version`), your OS, the command you ran, and the steps or a proof of concept.
Remove keys, tokens and anything from `.mcp.json` first. Valory maintainers acknowledge the report and keep you
informed while they work on a fix.

Only the latest release is supported with security fixes.

## In scope

- **The sandbox** (`olas_ocb_agent/sandbox.py`: Seatbelt on macOS, bubblewrap on Linux): the agent's shell reading
  what it's denied (keys, the real Connect token, session files, other services under `~/.operate`) or writing
  outside its own directory, including the workspace brief and skills.
- **The broker and its policy** (`broker.py`, `policy.py`): a call that moves funds or signs without the approval
  the session's `--approval` mode requires, a session cap (`--max-native-usd`, `--max-mech-usd`,
  `--max-value-calls`, the model budget) that can be exceeded, or a spend that isn't counted, for example across a
  crash or a `--resume`.
- **Key and secret handling:** a private key, the Connect token or a provider API key reaching the agent, the model,
  a session file or the output (`--output json`'s `error_detail` included).
- **Skill trust** (`skill_trust.py`): an untrusted `SKILL.md` that the agent is told to follow as instructions, or
  an agent that can trust a skill for itself.
- **`install.sh`:** running anything unpinned or unchecked, or replacing an executable that isn't olas-ocb-agent's.

## Out of scope

- What a model chooses to do within the caps and approvals you set (for example `--approval auto` with high caps),
  and ERC-20 amounts, which the broker does not cap (documented in the README and docs/DESIGN.md).
- Runs with `--sandbox none`, or on native Windows, where there is no sandbox.
- Bugs in Pearl Connect, the mech facilitator, the mech marketplace or a model provider: report those to their
  maintainers (for Valory's own services, the same private channel here is fine and we will pass it on).
