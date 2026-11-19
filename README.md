# olas-ocb-agent: a minimal agent for on-chain work with Pearl Connect

olas-ocb-agent is a minimal general-purpose agent harness for on-chain work with Pearl Connect. "OCB" stands for
OnchainBench, the benchmark for agents trusted with on-chain keys: the agent is measured there like any other entry.
It is open source, and the leaderboard lists it with its version and frozen digest, so anyone can rerun the exact
code that was scored.

**Based on mini-swe-agent's design.** The design comes from
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) by the SWE-agent team (Kilian A. Lieret, Carlos E.
Jimenez and contributors): bash as the only tool, actions in plain text and a linear history. olas-ocb-agent
re-implements that core (it doesn't depend on mini-swe-agent) and adds what on-chain work with Connect needs.

**The question it answers:** can a minimal harness, integrated with Pearl Connect and the Olas marketplace, match
Claude Code for on-chain, marketplace and unattended work? The harness isn't the gap (on the
[harness leaderboard](https://huggingface.co/spaces/joelniklaus/harness-leaderboard), mini-swe-agent and Claude
Code tie on the same open model). The design and its reasons are in [docs/DESIGN.md](https://github.com/valory-xyz/olas-ocb-agent/blob/main/docs/DESIGN.md);
the evidence comes from OnchainBench, whose run-1 analysis is fixed in advance in its
[METHODOLOGY.md](https://github.com/valory-xyz/onchainbench/blob/main/docs/METHODOLOGY.md).

## Install

Pearl Connect installs and launches the agent for you when you pick it; nothing needs to be on the machine first.
To install it yourself, from PyPI with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install 'olas-ocb-agent[facilitator]'           # the default model (Gemini through the mech facilitator)
uv tool install 'olas-ocb-agent[facilitator,litellm]'   # also --model litellm:<model>, with your own API key
```

Or with the install script (macOS, Linux, WSL2), fetched from the repo, without a checkout:

```bash
curl -LsSf https://raw.githubusercontent.com/valory-xyz/olas-ocb-agent/main/install.sh | sh
curl -LsSf https://raw.githubusercontent.com/valory-xyz/olas-ocb-agent/main/install.sh | sh -s -- 0.1.0   # a given release
```

Native Windows isn't supported by the script yet; on Windows, run it in WSL2.

- Installs `olas-ocb-agent[facilitator]==X.Y.Z` from PyPI (the script's release by default; enough for the default
  model) with `uv tool install`, on a uv-managed Python 3.12. It never uses or changes a system Python and edits no
  shell profile or `PATH`.
- uv: always the script's own copy of a pinned uv version (in `~/.local/share/olas-ocb-agent/bin`), never a uv found
  on `PATH`. When it's missing or another version, Astral's installer for the pinned version is downloaded and its
  sha256 checked against the one in the script before it runs; the installer in turn checks the uv it downloads
  (with `shasum` standing in for a missing `sha256sum`). It runs unmanaged: no `PATH` edit and no install receipt, so
  a uv you installed yourself keeps `uv self update`.
- Dependencies resolve as PyPI stood at the release (`--exclude-newer`, the date next to the version in the script).
  An `ocb-agent` or `pearl` already in the bin directory that isn't this tool's is never overwritten: the install
  fails instead (see [The `pearl` command](#the-pearl-command)).
- Another release: `sh install.sh X.Y.Z` (or `vX.Y.Z`), or `OCB_AGENT_VERSION`. `OCB_AGENT_SPEC` replaces the whole
  requirement, e.g. a GitHub archive
  (`'olas-ocb-agent[facilitator] @ https://github.com/valory-xyz/olas-ocb-agent/archive/<ref>.tar.gz'`) or a
  checkout (`'olas-ocb-agent[facilitator] @ file:///path/to/checkout'`). `XDG_DATA_HOME` and uv's own
  `UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, `UV_PYTHON_INSTALL_DIR` move what it installs.
- No prompts, and safe to rerun: when uv's olas-ocb-agent is already exactly that requirement (version, source and
  the facilitator extra) on a uv-managed Python 3.12 it changes nothing; otherwise it reinstalls in place (a custom
  `OCB_AGENT_SPEC` always does). On success the only line on stdout is the absolute path of `ocb-agent`, so a caller
  can launch it without `PATH`. On failure stderr gets exactly one `ocb-install: error: ...` line, naming a log with
  uv's details, and the exit code is non-zero. Progress goes to that log, and to stderr only at a terminal.

### The `pearl` command

The package also installs `pearl`, the agent's command for Connect's tools (`pearl tools`, `pearl <tool> ...`). It
has nothing to do with the unrelated `pearl` package on PyPI. If another tool's `pearl` already sits in uv's bin
directory, the install stops: install.sh's one error line names the file and says what to do. Uninstall the tool
that owns it (`uv tool uninstall <its package>`) or remove the file, then rerun.

## Quick start

```bash
ocb-agent --facilitator-deposit 50000000000000000           # your Pearl Connect workspace (found under ~/.operate)
ocb-agent --model 'mech:<tool>@<mech address>' --model-budget 0.05
ocb-agent --model litellm:anthropic/claude-sonnet-5-5       # bring your own key (needs the [litellm] extra)
ocb-agent --dir ~/some/project --model litellm:deepseek/deepseek-v4.1-flash   # any directory, no chain
ocb-agent --trust-skill <persistent_data>/.claude/skills/<name>/SKILL.md      # a skill as instructions
ocb-agent -p "what's in my safe?"                           # one turn, headless; --resume last to continue
ocb-agent --list-sessions                                   # the workspace's saved sessions, for --resume <id>
ocb-agent --help                                            # every flag, with its default
ocb-agent --version
```

- **The default model** (Gemini through the mech facilitator) needs a Pearl Connect workspace. It is paid from a
  prepaid balance that the session tops up from the Safe, up to `--facilitator-deposit` (base units:
  `50000000000000000` is 0.05 xDAI, about 50 calls; default 0, never). Each deposit is printed when it happens and
  listed in `/cost`. In a plain `--dir` without Connect, use `--model litellm:<model>`.
- **Approval** defaults to `--approval ask`: every value call shows a one-line summary (amounts in the token's units
  and USD, and what the session has left) and waits for y/N. A call nobody could answer (no terminal, end of input,
  the 10-minute timeout, Ctrl-C at the prompt) is refused as unanswered, not as declined.
- **Caps:** `--max-native-usd` (default $0.10) caps **native-token value only** (xDAI, ETH, ...): ERC-20 transfers and
  approvals are not capped, so in `--approval auto` they pass without a prompt. `--max-mech-usd` (default $1.00)
  caps mech payments. Both are per session, converted per chain. Raw digest signing is off unless
  `--allow-sign-message`. A call a rule blocks is printed as `[blocked] ...`.
- **Session cost limit** (`--cost-limit`, $2 by default, for API models): it is checked before each model call, so the
  call that crosses it finishes and the turn stops before the next one. The spend is restored on resume, so the
  session goes on only with a higher `--cost-limit`; or start a new one.
- **Skills:** a workspace `SKILL.md` is followed as instructions only if its SHA-256 is trusted; any other is shown
  to the agent as unverified data. You trust one with `--trust-skill <path>` (kept outside the sandbox). The
  package itself pins none: **a launcher** such as Pearl passes `--trusted-skill-hash <sha256>` once for each skill it
  ships (repeatable, this session only).
- **Plain project folders** (`--dir`): `AGENTS.md` and `CLAUDE.md` are read-only to the agent, since the next session
  reads them as your brief; `--allow-brief-edits` lets it edit them. A Connect workspace's brief is always read-only.
  On Linux, a brief that is a symlink can't be kept read-only, so the agent refuses to start there unless
  `--allow-brief-edits`.
- **Caches:** the user's `~/.cache` and `~/Library/Caches` are read-only to the agent; its tools cache under
  `.ocb-agent/cache` (`XDG_CACHE_HOME`, uv, pip, pre-commit, Go, Yarn, Deno, Playwright). A tool that ignores these
  variables can't write its cache on macOS.
- **REPL commands:** `/compact`, `/cost` (model spend, each limit and what's left, deposits, blocked calls) and
  `/exit`. Ctrl-C kills the running command and cancels an approval prompt (on native Windows, where stdin can't be
  polled, the cancelled prompt's reader thread stays waiting and may take the next line). `--resume` prints a recap.
  Status lines (deposits, `[blocked]` calls, the recap) go to stderr. Output has colours only at a terminal.
- **Errors** (a mistyped flag, a model provider that fails, a missing extra) print one `ocb-agent: error: ...` line
  and exit non-zero.

## What it is

1. **The harness** (`olas_ocb_agent/`): mini-swe-agent's core (bash only, plain-text actions, linear history) plus
   chat turns, compaction, a persistent shell, resumable sessions and streaming. Model backends: Gemini through the
   Olas mech facilitator (the default, signed through Connect, no API key), marketplace mechs through Connect, or
   LiteLLM with your own key.
2. **Connect access:** a broker that speaks Connect's MCP and REST API, a shim `.mcp.json`, and the `pearl` command.
   Connect's skills and `signer_client.py` work unchanged.
3. **Safety:** an OS sandbox (Seatbelt on macOS, bubblewrap on Linux and WSL2) with keys and the real token
   unreadable (all of `~/.operate` but the Connect workspace) and writes confined to the agent's own directory
   (`<workspace>/ocb-agent/`; the workspace brief and skills can't be rewritten), plus session approval and spend
   caps in the broker, in front of Connect's guardrail. Session files (history and the spend ledger `--resume`
   restores) live outside the workspace, in `$XDG_STATE_HOME/olas-ocb-agent/sessions/<sha256 of the workspace path>/`
   (by default under `~/.local/state`), where the sandboxed agent can neither read nor write them. With
   `--sandbox none`, or on native Windows where there is no sandbox, nothing stops the agent's shell from editing them.
4. **On OnchainBench** the agent is a CLI arm like every other harness (`olas-ocb-agent:M`, `olas-ocb-agent-mech:M`):
   the bench runs `ocb-agent -p --output json` in its own process (see [Headless runs](#headless-runs-benchmarks)).
   The agent doesn't depend on the bench.

Formal checking: a Quint model of the broker policy, spend across crashes and the marketplace nonce in
[quint/](https://github.com/valory-xyz/olas-ocb-agent/blob/main/quint/README.md).

## Chains

Any EVM chain in principle, 9 EVM chains out of the box (Ethereum, Gnosis, Polygon, Arbitrum, Optimism,
Base, Celo, Mode, Robinhood). The chain comes from Connect (`wallet_info`); `olas_ocb_agent/chains.py` holds only a
rough USD price per chain for the broker's caps (`--native-usd <chain>=<price>` for others). Mech payment assets are
read from the mech itself, so every `mech_request` names its mech.

## Headless runs (benchmarks)

`-p` runs one turn and exits. By default the turn is unattended: the system prompt says nobody will reply, and the turn
ends only on a `DONE:` reply. With `--attended` a `-p` turn behaves like a REPL turn: it may end with a question, and
`-p "<answer>" --attended --resume <session id>` continues the same session.

Each `-p --resume` turn is a new process, as with other CLI harnesses. The history, the counts, the spend ledger
(caps and approvals) and the shell's working directory carry over. Shell variables, functions and background jobs
don't: each turn starts a new shell.

```bash
ocb-agent --dir <workspace> --model 'litellm:openrouter/<model>?provider=<provider>' --approval auto \
  --output json -p "<task>"
```

- **Approvals** come from `--approval` (`auto` or `deny` for headless runs). JSON mode has no interactive approval
  channel.
- **`--output json`** prints one JSON object on stdout, also when the turn fails or times out:
  - the turn: `reply`, `session_id`, `done` (the turn ended on `DONE:`), `error` and `error_detail` (truncated,
    secrets redacted). `error` is null on success, else `"model"` (a model budget or cap), `"provider"` (the model
    API or the network failed, after retries), `"harness"` (anything else, including a bad `--resume`),
    `"timeout"` (`--turn-seconds`) or `"terminated"` (SIGTERM). Any error exits non-zero.
  - session totals, restored across `--resume`: `model_calls`, `prompt_tokens`, `completion_tokens`, `cached_tokens`
    (0 where the provider doesn't report them), `cost_usd` (the agent's own figure: what OpenRouter billed, else
    tokens x price), `cost_sources` and `compactions`.
  - this turn only: `steps` (model calls), `policy_refusals`, `broker_refusals` and `tools_listed`, plus
    `session_file` and `events` (the session events this turn logged; a new session's start with the system
    prompt).
  - timing, on every event in `events` and in the session file: `t`, seconds from the turn's start to when the
    event was logged (0 for the system prompt). A model's reply (an assistant `message`, or a `compaction`) also
    has `dur_s`, from the request sent to the reply complete, and `ttft_s`, the time to the first token (backends
    that don't stream deliver the reply in one piece, so there it is about `dur_s`). A command's output (the
    observation `message`) has `dur_s`, from the command sent to its return code. An event's work started at
    `t - dur_s`, so the time between two steps is the next one's start minus the previous one's `t`. The timing
    sits on the event beside `message`, never in what the model is sent.

  The banner and progress go to stderr, without colour.
- **Limits:**
  - `--step-limit N` (model calls per turn, default 30) and `--cost-limit USD` (per session, default 2);
  - `--max-value-calls N` (default 20; 0 blocks value calls; `none`: no limit), and `--max-native-usd` and
    `--max-mech-usd` (`inf`: no cap);
  - `--turn-seconds N`: the turn stops itself after N seconds, kills its running command and still reports. SIGTERM
    does the same.

  Negative or NaN limits are refused.
- `--allow-sign-message` lets the agent request raw digest signatures (off by default).
- `--model-price JSON` registers a LiteLLM model_cost entry (numeric `input_cost_per_token` and
  `output_cost_per_token` required) for the model in this process. It feeds the cost estimate and `--cost-limit` when
  the provider reports no charge.
- **Workspace and shell settings:**
  - `--sessions-dir`;
  - `--agent-subdir`: the agent's directory in a Connect workspace, default `ocb-agent`;
  - `--trusted-skill-hash SHA256`: a skill file trusted as instructions for this session only (64 hex characters,
    repeatable). This is how a launcher such as Pearl trusts the skills it ships;
  - `--shell-env KEY=VALUE`: a variable for the agent's shell only, e.g. a logging web proxy (repeatable; `PATH`,
    `HOME`, `BASH_ENV`, `ENV`, `LD_*`, `DYLD_*` and the variables the harness sets for the shell are refused).

## Development

```bash
git clone https://github.com/valory-xyz/olas-ocb-agent && cd olas-ocb-agent
uv venv .venv -p 3.12 && VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
scripts/check.sh
```

- **Every new feature or fix comes with a test:** `pytest` fails below **100% line coverage** of `olas_ocb_agent/`.
- **ruff** check and format must pass (line length 120; rules in pyproject). A `noqa` needs a one-line reason;
  `# pragma: no cover` only with a reason.
- **The gate:** `scripts/check.sh`, also run in CI (`.github/workflows/check.yml`).
- Tests run against `tests/mock_connect/`, the agent's own Connect test double (a copy of OnchainBench's mock).

## Contributing and security

Issues only: pull requests from outside Valory are closed, because Valory develops the agent with AI agents. Fork it to
build your own. See [CONTRIBUTING.md](https://github.com/valory-xyz/olas-ocb-agent/blob/main/CONTRIBUTING.md).
Report a vulnerability privately, never in an issue: see
[SECURITY.md](https://github.com/valory-xyz/olas-ocb-agent/blob/main/SECURITY.md).

## Documents

- [docs/DESIGN.md](https://github.com/valory-xyz/olas-ocb-agent/blob/main/docs/DESIGN.md): the current design and its
  reasons: principles, architecture, the loop, models, Connect access, the sandbox, the policy, known gaps and the
  threat model.
- [quint/](https://github.com/valory-xyz/olas-ocb-agent/blob/main/quint/README.md): the Quint model and its invariants.
- [olas_ocb_agent/interim/](https://github.com/valory-xyz/olas-ocb-agent/blob/main/olas_ocb_agent/interim/README.md):
  the stopgap code that goes away once Connect or the mech marketplace does the job itself, and what replaces each
  module.
