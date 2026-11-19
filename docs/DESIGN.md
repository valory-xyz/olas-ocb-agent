# olas-ocb-agent: design

olas-ocb-agent is a minimal, general-purpose agent harness for on-chain work with Pearl Connect, built on the design
of [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent). It is measured on OnchainBench like any other
entry. This is
the current design and the reasons for it.

## What it expects from Connect and the marketplace

Some of the agent is a stopgap for features that belong in Pearl Connect or the Olas mech marketplace. Once they
exist, the stopgap goes (the code is in `olas_ocb_agent/interim/`):

| Upstream change | What it replaces here |
|---|---|
| chat models served as marketplace mechs (chat-completion tools), reachable through Connect | `FacilitatorModel`, the interim default model |
| every off-chain request, model calls included, paid through Connect, so Connect is the only payer per Safe | `MarketplaceGate` (nothing collides with a single payer) |
| the facilitator's mechs as full marketplace mechs (published metadata and tools, the standard off-chain request path) | the facilitator's payment-asset table, and its separate pre-deposit |
| scoped per-session tokens in Connect, with caps Connect enforces | the broker's role as the only place session caps live |
| transaction simulation, a structured intent check, and chain read/decode tools inside Connect | the fallback list of read-only tools, and the gap in ERC-20 amounts |
| skill docs that match Connect's real tool schemas | nothing; fewer calls with unknown parameters for the `pearl` CLI to refuse |
| a supported-chains list published by Connect | the price table in `chains.py`, as anything but a fallback |

## What it has to do

- **Work on any task a Pearl user brings:** on-chain actions through the user's agent Safe, research, files, and
  chat over many turns and long sessions.
- **Be safe with real money:** a prompt-injected or confused agent must not be able to move funds the user didn't
  approve, or read the keys or the signer's token.
- **Work with any model,** including ones bought from the Olas marketplace with no API key.
- **Stay small enough to audit:** a wallet-holding agent is only as trustworthy as the code a user has to trust.

## Principles, and why

- **A small trusted core; the model handles the diversity.** On the
  [harness leaderboard](https://huggingface.co/spaces/joelniklaus/harness-leaderboard) (nine harnesses, the same
  open model and effort, Terminal-Bench 2.1 + DeepSWE), mini-swe-agent (bash only, about 0.66) is statistically
  tied with Claude Code and Hermes, and beats Claude Code on the weakest model. Tool-selection accuracy falls as the
  tool count rises ([RAG-MCP](https://arxiv.org/abs/2505.03275)), and code execution beats loading many tool
  schemas ([Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp), Cloudflare's Code Mode). So
  the agent keeps mini-swe-agent's core and adds only what Pearl needs: chat, long sessions, Connect access and
  safety.
- **What matters on Pearl-style tasks is configuration, not code**. With a Pearl system prompt, a clearer
  format-error message and sane limits, unmodified mini-swe-agent comes close to this agent on Pearl-style tasks.
  Its value is therefore in product features (chat turns, a persistent shell,
  compaction, sessions, the broker, the sandbox), not in a cleverer loop.
- **No plugin or skill stores.** That's where wallet agents lose money: ClawHub had 1,184 malicious packages at
  peak, mostly stealers aimed at seed phrases, and MCP tool poisoning and registry rug pulls are documented attack
  classes. The agent loads only code its vendor ships and pins; Connect's skills are read as instructions, and an
  untrusted skill has to be trusted explicitly (`--trust-skill`).
- **One Valory primitive, `pearl`:** wallet and marketplace through Connect. Chain reads, ABI resolution and
  decoding belong in Connect, next to the simulation and intent check it should gain, so the agent ships no chain code and
  every harness gets the same tools.
- **Three independent checks on each transaction:** the session policy here (ask or cap), Connect's guardrail, and
  Safe isolation: each agent has its own Safe, separate from the user's Pearl wallet, so it can lose at most what
  was moved into it. A fourth, simulate → diff → check against a declared intent, belongs in Connect.

## Architecture

```
            harness process (not sandboxed)                       sandboxed bash
 ┌───────────────────────────────────────────────┐   ┌─────────────────────────────────┐
 │ reads <ws>/.mcp.json  (real token, in memory) │   │ cwd <ws>/ocb-agent/             │
 │ Broker 127.0.0.1:<port>  (broker token)       │◄──┤  .mcp.json  (shim: broker URL)  │
 │   same MCP + REST API as the connect server   │   │  pearl wallet_info              │
 │   Policy: reads free, value calls ask/caps    │   │  python script → signer_client  │
 │ MechModel ──► connect server (model calls)    │   │  cat ../.mcp.json  → EPERM      │
 └────────────────────┬──────────────────────────┘   └─────────────────────────────────┘
                      ▼
          connect server :8716 (keys, guardrail, signing, audit log)
```

**Pearl Connect** is the local server the agent works through. It custodies the agent EOA key, signs and
broadcasts, and applies a guardrail: delegatecall and self-calls are never allowed, plus operator rules
(`unrestricted` or a per-chain whitelist), integrity-checked so a hand-edit resets them to safe defaults. It exposes
MCP tools (`wallet_info`, `safe_transaction`, `send_transaction`, `preflight_transaction`, `transaction_status`,
`sign_message`, `mech_tools`, `mech_request`, `mech_result`) and REST routes for scripts (`signer_client.py`), and it
already writes the workspace files harnesses read (`AGENTS.md`, `CLAUDE.md`, skills, `.mcp.json`). The agent
consumes those same files, so Connect needs nothing agent-specific.

**Why the broker and the sandbox exist:** Connect's token sits in the workspace's `.mcp.json`, and
`signer_client.py` reads it from any script the agent runs. A harness's "don't read it" rule is an instruction, not
a control: a prompt-injected agent can call the signer directly and skip whatever the harness would have asked the
user. So the real token stays in the harness process, the agent's shell can't read it, and everything the shell
does reaches Connect through a broker that applies the session's policy. Scoped per-session tokens in Connect
would make this protection available to every harness.

## The harness loop (`agent.py`, `shell.py`)

**Kept from mini-swe-agent:** bash is the only tool; actions are parsed from one fenced block in plain text, so any
model works without a tool-calling API; the history is linear, so the trajectory is exactly what the model saw;
there are no tool schemas to maintain. The block closes on a ```` ``` ```` line of its own (a ```` ```lang ```` line
inside it, such as a heredoc holding a fenced example, opens a nested block); anything after it is dropped unread,
and the observation says so.

**Added, each as small as it can be:**

| Need | How | Why |
|---|---|---|
| Chat | a reply with no bash block ends the turn and goes to the user | Pearl is conversational, not one-shot |
| Shell state | one persistent bash (cd, exports, venvs persist); stdin is `/dev/null`; each command and its end marker (a fresh random nonce) go to bash as one line, so the command can't read the nonce or end its own output early; a reported cwd that isn't a directory is ignored; on timeout, kill the process group and restart in the same cwd (or the starting one, if that is gone); once closed, the shell stays closed and a running turn ends | real work needs state between commands; a hung command must not hang the agent; a turn its caller gave up on must not act after the session closed |
| Long sessions | compaction: past 75% of the context budget, the model summarises the old messages; the last N stay verbatim | sessions outlive a context window |
| Resume | an append-only JSONL session (messages, compactions, cwd, everything spent, the approved request ids, the model calls and tokens counted) in `$XDG_STATE_HOME/olas-ocb-agent/sessions/<sha256 of the workspace path>/` (default `~/.local/state`), outside the agent's directory and denied to the sandbox (read and write, every workspace's sessions); without a sandbox (`--sandbox none`, native Windows) this protection doesn't hold; `--resume last` replays it with the same caps | a crash or restart must not reset the caps or lose the work, and the agent must not edit its own ledger or history |
| Streaming | `on_token`, `on_action` and `on_observation` callbacks | the CLI prints them; Pearl's UI would render them |
| Limits | steps per turn, cost per session, a model budget in the mech's payment units | runaway loops and runaway spend |
| Context | Connect's brief (`AGENTS.md`), a skills index (name, description, path; read on demand), the harness rules | the model learns Connect from Connect's own docs, without loading every skill |

## Models (`model.py`; interim: `interim/facilitator.py`, `interim/marketplace_gate.py`)

- **`MechModel`, the target default:** each completion is bought from a marketplace mech through Connect's
  off-chain flow (`mech_request` with `legacy_on_chain=false`: a signed digest drawing on the prepaid balance; no
  transaction, no API key). The prompt is the messages list as JSON. The session has a model budget in the payment
  token's base units, and each request's `max_payment` is capped at what remains, so the mech refuses before
  payment rather than overspending. Pending request ids are resumed with `mech_result`, never resent. Model calls go
  from the harness to Connect directly, not through the broker: the user sets a budget instead of approving every
  completion.
- **`FacilitatorModel`, the interim default:** the marketplace has no general chat mechs yet; the only usable
  models are Gemini 2.5 and 2.0 behind the mech facilitator's `/mech/chat` route, at $0.001 per call on Gnosis,
  Base, Polygon and Optimism. The agent signs those requests through Connect's `sign_message` (a byte-for-byte port
  of Pearl's agents' signing, tested against the same vectors), so it never holds a key. A pre-deposit is sent by
  the Safe, only up to the session's `--facilitator-deposit` allowance. Retries replay the same signed body, so a
  call is never charged twice.
- **`MarketplaceGate`:** the facilitator and the agent's own mech requests share one marketplace nonce per Safe. A
  request signed at a nonce the facilitator still holds makes its settlement fail, and the facilitator pauses the
  Safe (this happened in live use; in simulation, every ungated session pauses within about half an hour). The gate
  serialises the two paths within the process. The real fix is one payer: every request through Connect, with
  the models offered as full chat mechs. Then `FacilitatorModel` and `MarketplaceGate` (about 540 of the
  agent's 4,280 lines) are deleted and only `MechModel` remains.
- **`LitellmModel`:** bring-your-own-key, with token streaming. **`ScriptedModel`:** tests and oracle runs.

## Connect access (`broker.py`, `pearl_cli.py`, `workspace.py`)

- **A shim `.mcp.json`.** The agent works in `<ws>/ocb-agent/` (the bench's `agent/` in OnchainBench trials), whose
  `.mcp.json` points at the broker with a broker-only token. `signer_client.py` walks up from the cwd to the first
  `.mcp.json`, so Connect's scripts and skills work unchanged and find the shim first.
- **The `pearl` CLI** maps every MCP tool to a command (`pearl tools`, `pearl help <tool>`,
  `pearl <tool> '{json}'` or `key=value`). Tool descriptions come live from Connect, so new tools appear without a
  release. It accepts `--key=value` and refuses keys that aren't in the tool's schema, with a did-you-mean hint,
  because models write CLI flags and Connect silently ignores unknown parameters.
- **The broker** speaks Connect's MCP and REST API. Reads pass; every value call (`safe_transaction`,
  `send_transaction`, `mech_request`, `sign_message` and their REST forms) passes the session policy first. A tool
  passes without asking only if Connect flags it read-only (MCP `readOnlyHint`); until Connect sends flags, a
  fallback list of its five read-only tools applies, and a new unflagged tool needs approval.

## The sandbox (`sandbox.py`)

- **macOS (Seatbelt, the mechanism Claude Code uses there):** reads are denied for the real `.mcp.json`,
  `.codex/config.toml`, all of `~/.operate` (Pearl's keys and wallets, and every service's `config.json` and
  `keys.json`) except the Connect workspace inside it, `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`, `~/.netrc`
  and the keychains, and for the session directory; writes are allowed only in the agent's own directory
  (`<workspace>/ocb-agent/`, not the whole workspace) and temp dirs. A denied file, and every directory
  above it, can't be renamed or replaced, so the denial can't be dodged with `mv .mcp.json x && cat x`.
- **`~/.operate` is denied as a whole,** with the workspace read again on top (what is denied inside it, like the
  real `.mcp.json`, stays denied): a list of its secret files always missed some.
- **The workspace root is read-only:** everything under a Connect workspace root except the agent's directory (the
  brief, the skills, the real `.mcp.json`, any new file) can't be written, and neither the root nor a directory above
  it can be renamed, even when the root sits under a temp dir the sandbox may otherwise write. Writes to `AGENTS.md`,
  `CLAUDE.md`, `.agents/` and `.claude/` are also denied explicitly: the next session of any harness reads the brief
  as the owner's and the skills as instructions. In a plain directory (no Connect workspace) the agent's directory
  is the root, and its `AGENTS.md` and `CLAUDE.md` are read-only too by default, since the next session loads them as
  the owner's brief; `--allow-brief-edits` lets the agent write them when editing them is the task (the prompt says
  which applies). Seatbelt denies a brief both as named and as resolved, so a symlinked brief can't be removed and
  replaced. On Linux a mount can't cover a link, so a symlinked brief stops the start (unless `--allow-brief-edits`),
  and a brief file that doesn't exist yet can't be covered, so there the agent could create one.
- **The harness never writes through the agent's links:** the shim `.mcp.json`, `.gitignore` and the `pearl` script
  are written as new files renamed into place (a symlink there is replaced, not followed), a symlinked `.ocb-agent/`
  or `bin/` is refused, and so is an agent directory that is (or sits under) a symlink. The agent's directory itself
  can't be removed, renamed or replaced from the sandbox. Commands reach the shell on its stdin (quoted for `eval`),
  so the harness writes no script file at all.
- **The user's caches are read-only as a whole** (`~/.cache`, `~/Library/Caches`): code outside the sandbox runs
  from them later (uv hard-links installed packages, pip installs cached wheels, pre-commit runs its hook envs,
  Homebrew pours downloaded bottles), and a list of the dangerous ones always missed some. The agent's tools get
  caches of their own under `.ocb-agent/cache` (`XDG_CACHE_HOME`, `UV_CACHE_DIR`, `PIP_CACHE_DIR`,
  `PRE_COMMIT_HOME`, `GOCACHE`, `YARN_CACHE_FOLDER`, `DENO_DIR`, `PLAYWRIGHT_BROWSERS_PATH`), git-ignored there. A tool
  that ignores these variables can't write its cache on macOS (it would use `~/Library/Caches`).
- **Closing the shell kills everything it started:** its process group and every descendant, including those that
  left the group with `setsid` (on Linux the sandbox's PID namespace does this too).
- **Linux (bubblewrap; Windows through WSL2):** the filesystem is read-only except the agent's directory and temp
  dirs; denied files are covered by `/dev/null` and denied directories by an empty read-only tmpfs (a covered
  path is a mount point, so it can't be moved either), with a re-allowed workspace inside one bound again on top. Whatever must not change under a writable temp dir (the
  workspace root, denied paths) has its topmost directory there bound read-only, and the agent's directory bound
  writable on top: stacked mounts cover files that don't exist yet and block the rename. The shell gets its own PID
  namespace and dies with the agent.
- **The network stays open,** because the agent needs the web and the broker. Without a sandbox available, the agent
  says so on start (`NO SANDBOX`), and the broker's policy is the only guard.

## The session policy (`policy.py`)

The policy sits in front of Connect's guardrail, never instead of it.
- **Modes:** `ask` (the default: every value call shows a one-line summary of who is paid what, in the token's
  units and USD, with what the session has left after it, and waits for y/N), `auto` (allowed within the caps) and
  `deny`.
- **Nobody answering is not a no:** with no terminal to ask, at end of input (Ctrl-D), after the timeout, or after a
  Ctrl-C at the prompt, the call is refused by its own rule (`no-approver`, `input-closed`, `approval-timeout`,
  `approval-cancelled`), never as "the user declined". After Ctrl-C the prompt stops reading stdin before the REPL
  reads its next line, and any prompt the stopped turn opens afterwards is cancelled at once, so on macOS and Linux
  there is one reader. On native Windows stdin can't be polled: the prompt reads in a thread, which a cancel leaves
  waiting, so it may take the user's next line.
- **Every refusal is one line on screen** (`[blocked] <tool>: <rule>`), and `/cost` counts them.
- **Caps per session, in USD converted per chain**: native-token value (default $0.10; ERC-20 transfers and approvals
  are not counted, and auto mode says so at start), the sum of declared mech
  `max_payment` (default $1.00), and the number of value calls (20). Raw digest signing is off unless
  `--allow-sign-message`. Every `mech_request` must name its mech and an
  explicit `max_payment`, so its payment asset and price are known. A negative `value` or `max_payment` is refused
  (rule `negative-amount`), and so is a non-integer one or one of 2**256 or more (`bad-amount`), so counted spend
  only ever grows. The spend is
  saved the moment it is counted, so `--resume` continues it and a crash can't reset it.
- **One deadline per broker call**: an `ask` approval is refused (`approval-timeout`), and the gate gives up, 10 minutes after the
  call arrives; meanwhile the shell command doesn't time out, so the agent never sees a timeout for a call that is
  then sent. A killed command's pending calls are refused.
- **Retries** with an already-approved `request_id` and identical arguments pass without a new prompt or spend,
  matching Connect's idempotency; the same id with different arguments is a new call.
- **Refusals name their rule,** and the prompt tells the agent not to work around one (for example by switching
  from the Safe to the EOA).

## Chains (`chains.py`)

Any EVM chain in principle, 9 EVM chains out of the box (Ethereum, Gnosis, Polygon, Arbitrum, Optimism, Base, Celo,
Mode, Robinhood). The chain always comes from Connect, never from the agent. `chains.py` is data only (the native
symbol, a rough USD price and the default mech payment asset per chain), used to convert the USD caps; other chains need `--native-usd`. Mech payment assets are read from the mech
itself. A Connect that publishes its own chain list would make the table a fallback.

## Known gaps, and what closes them

| Gap | Bounded today by | Closed by |
|---|---|---|
| ERC-20 amounts aren't capped by the broker (only native value is priced); models do mis-encode amounts | agent Safe isolation | an intent check in Connect |
| Connect's `auto_deposit` top-ups aren't counted by the broker or the model budget | they stay in the agent's Safe and prepaid balance | scoped caps in Connect |
| The facilitator pre-deposit bypasses the broker's prompt and caps, and is native-only (works on Gnosis only) | the `--facilitator-deposit` allowance, default 0; each deposit is shown on screen, saved at once and listed in `/cost` | facilitator mechs as full marketplace mechs |
| The nonce gate protects one process, not several on the same Safe | running one agent process per Safe | one payer through Connect |
| Connect flags no tool read-only | a fallback list of five read-only tools | `readOnlyHint` in Connect |
| The guardrail ships as `unrestricted` | the broker's policy | the whitelist default, measured in OnchainBench |

## Threat model

| Threat | A harness on Connect today | olas-ocb-agent |
|---|---|---|
| Injected text tells the agent to send funds | the model must refuse; the guardrail may block | the model must refuse; **plus** the broker asks or caps; plus the guardrail |
| The agent reads the token and calls the signer directly | possible from any script (`signer_client.py` does it by design) | blocked by the sandbox; the shim token reaches only the broker |
| The agent reads key files | they're encrypted; the password isn't on disk for the agent | blocked by the sandbox as well |
| A lost response, then a double spend | Connect's idempotency via `request_id` | the same, and the broker recognises the retry |
| Runaway spend on model calls | n/a (subscription) | the model budget; the mech refuses above what remains |
| The agent writes outside the workspace | allowed, subject to the harness's permission prompts | blocked by the sandbox |
| The agent rewrites the workspace brief or skills for the next session | allowed | blocked by the sandbox (writes go to `<workspace>/ocb-agent/` only) |
| The agent edits its session file to reset its caps or plant a user approval for `--resume` | n/a | the session directory is outside the workspace and unreadable and unwritable in the sandbox |

## Size and dependencies

About 4,280 lines (`wc -l $(find olas_ocb_agent -name '*.py')`), of which about 580 are interim: everything in
`olas_ocb_agent/interim/` goes away once Connect or the mech marketplace does the job itself, and its README maps
each module to the change that removes it and what replaces it. Standard library only, apart from the optional extras:
`litellm` (bring-your-own-key) and `eth-abi`/`eth-utils` (the interim facilitator backend). A Quint model of the
policy, spend across crashes and the marketplace nonce is in [../quint/](../quint/README.md).
