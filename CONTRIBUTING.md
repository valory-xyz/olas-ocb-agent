# Contributing to olas-ocb-agent

**Contributions are welcome as issues only. Pull requests from outside Valory are closed without review,**
automatically, with a comment pointing here.

Valory develops olas-ocb-agent with AI agents. A clear issue is the fastest way to get a bug fixed or a feature
built: the agents can turn an agreed issue into a tested change faster than an outside PR can be reviewed,
reworked and brought to the repo's bar (100% test coverage, the Quint model, the frozen versions OnchainBench
scores). Closing a PR says nothing about its quality. If it fixes something, open an issue that describes the
problem, and link the PR as a reference.

## What to open

- **A bug:** what you ran (version, from `ocb-agent --version`; command, model, chain), what happened, what you expected, and the session
  file or log if you can share it. Remove keys, tokens and anything from `.mcp.json` first.
- **A feature or change:** the problem it solves for someone running the agent, and how you'd know it works.
  Say why the change belongs in a *minimal* harness: the agent stays small on purpose (docs/DESIGN.md,
  "Principles, and why").
- **A security problem:** don't open an issue. Report it privately through GitHub's private vulnerability
  reporting on this repo (the *Security* tab); [SECURITY.md](SECURITY.md) says how, and what's in scope.

## How Valory decides

Valory maintainers read every issue and reply with a decision and the reason: accepted (and then built, usually
by Valory's agents), turned down, or needs more information. The criteria are the agent's purpose: a minimal,
safe harness for on-chain work through Pearl Connect. A change that makes
the agent safer or more capable on that work, without making it much bigger, is the kind that gets in.

## Building your own

The code is Apache-2.0: fork it and build what you like. To compare your version with others, enter it in
OnchainBench as your own harness: its [CONTRIBUTING.md](https://github.com/valory-xyz/onchainbench/blob/main/CONTRIBUTING.md),
"Harnesses", explains how. Name it so it can't be mistaken for olas-ocb-agent.
