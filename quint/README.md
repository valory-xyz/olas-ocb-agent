# olas-ocb-agent: Quint model

A Quint model of the broker's session policy, spend across crashes and `--resume`, the marketplace nonce shared by
the facilitator and Connect (`MarketplaceGate`), the deadline of a broker call against the shell's, and the turn's
step limit.

## How to run

Quint 0.33, default (Rust) backend.

```bash
quint/check.sh            # every check below, about 2 minutes (N, STEPS, SEED override the defaults 60000, 50, 11)
quint typecheck quint/ocb_agent.qnt
quint run quint/ocb_agent.qnt --main=main --invariant=allInvariants --max-samples=60000 --max-steps=50 --seed=11
quint run quint/ocb_agent.qnt --main=save_after_command --invariant=inv1_caps --mbt --verbosity=3   # a counterexample
```

## Modules

`ocb_agent` is the model, parametric in one switch per design choice. The other modules set the switches:

| Module | What it is | Expected |
|---|---|---|
| `main` | the shipped design, after the fixes below, plus assumption A1 | all invariants hold; every witness reachable |
| `one_payer` | the target design: every request, model calls included, paid through Connect's own nonce allocator; no facilitator, no gate | all invariants hold |
| `save_after_command` | **finding F1:** spend saved only after each command and model reply (earlier code) | `inv1_caps` violated |
| `kill_keeps_calls` | **finding F2:** a killed command's waiting broker calls stay and can be approved and sent (earlier code) | `inv6_noLateSend` violated |
| `kill_misses_queued` | **finding F4:** a call queued behind a slow send, past the kill check, is sent after the kill (earlier code) | `inv6_noLateSend` violated |
| `floor_lost_on_crash` | **finding F5:** the gate's pending floor in memory only, lost on a crash (earlier code) | `inv4_noCollision` violated |
| `late_settle_after_drop` | **assumption A1:** a Connect request settles after the gate gave up on it | `inv4_noCollision` violated |
| `no_gate` | no gate: the facilitator and Connect sign the same nonce and the facilitator pauses, as seen in live use | `inv4_noCollision` violated |
| `default_asset` | **finding F6:** mech requests without `priority_mech` priced in the chain's default asset (the design before `priority_mech` was required) | `inv1_caps` violated |
| `connect_fallback` | the residual assumption: Connect falls back from the named mech to one paid in another asset | `inv1_caps` violated |
| `no_value_guard` | the value guard off: a value tool Connect flags read-only is trusted | `inv1_caps` violated |
| `paid_uncounted` | **finding F7:** a paid model call whose follow-up fails isn't counted | `inv3_model` violated |
| `no_hold` | **finding F8:** no hold, the shell times out under a pending approval | `inv6_noLateSend` violated |
| `hold_uncapped` | **finding F9:** an uncapped hold lets broker calls keep a command alive | `inv6_cmdBounded` violated |
| `negative_counted` | **finding F3:** a negative `value` or `max_payment` is counted, lowering the counters | `inv1_caps` violated |

## What is modelled

- **Policy** (`olas_ocb_agent/policy.py`): modes ask/auto/deny; caps on native USD, mech USD and value calls; `priority_mech`
  required; the mech's own asset prices `max_payment`; read-only tools from Connect's flags, never a value tool;
  approved `(request id, args)` retries. Connect pays at most once per request id, and never above `max_payment`.
  Real money leaving the safe is tracked apart from the counters.
- **Spend persistence**: process counters, the last saved state (with the gate's pending floor), crash and resume.
- **Models:** marketplace model calls against the model budget, with follow-ups that fail after payment; the
  facilitator pre-deposit within its allowance.
- **Marketplace nonce:** facilitator rows signed at max(its next slot, the on-chain nonce), Connect requests at the
  on-chain nonce (with one payer: Connect's own allocator), settlement in nonce order, pause on a used slot; the
  gate's floor, its waits and the floor drop after 2 failed waits; expired Connect requests.
- **Deadlines:** a command, broker calls arriving during it, approvals, a call queued behind the upstream lock and
  its send, the call deadline (4 ticks), the shell deadline (2 ticks) that fires on time unless a call holds it, the
  capped hold, the kill.
- **Turn:** the step limit and nudges.

**Abstractions:** cents, caps scaled down (native 10, mech 40, 4 value calls, model budget 12) so random runs reach
them; two mechs (one native, one stablecoin at 10 cents per base unit); integer ticks.

**Not modelled:** the sandbox (tested in `tests/test_units.py`), ERC-20 amounts (not capped by the broker; an intent
check in Connect would close that), Connect's `auto_deposit` top-ups (not counted by the broker; caps on scoped
Connect tokens would), stablecoin symbol spoofing, other processes on the same safe.

## Invariants

| | Invariant | Holds in `main` |
|---|---|---|
| `inv1_caps` | counters and real spend within the caps, and counters never negative, for the whole session (across crashes) | yes |
| `inv2_retry` | an approved `(id, args)` replayed changes no counter | yes |
| `inv3_model` | real model spend ≤ the budget; deposits ≤ the allowance | yes |
| `inv4_noCollision` | the facilitator is never paused | yes (with A1) |
| `inv6_turn` | steps ≤ the limit, nudges ≤ 3 | yes |
| `inv6_noLateSend` | no call is sent after its command was reported timed out | yes |
| `inv6_cmdBounded` | a command runs at most the shell deadline plus the call deadline (+1 tick) | yes |
| `inv8_flows` | every money flow has one known source and one known destination | yes |

## Findings

- **F1 (fixed): a crash mid-command lost the session's spend.** The broker counts spend during a command, but the
  session file got it only after the command (or the model reply). A crash in between restored older counters on
  `--resume`, so the session could spend past its caps. **Fix:** the policy saves the moment
  it counts (`Policy.on_record`), and a model call that fails after paying saves before the error goes up
  (`Agent._query`). Tests: `test_broker_spend_is_saved_at_once_and_survives_a_crash`,
  `test_a_failed_paid_model_call_is_saved`. Regression module: `save_after_command`.
- **F2 (fixed): a killed command's call could still be sent.** The capped hold (F9's fix) lets the shell
  kill a command while a second broker call still waits for approval; a later "y" sent it.
  **Fix:** the shell tells the broker when it kills a command (`Broker.command_killed`), and calls not yet sent are
  refused. Test: `test_calls_of_a_killed_command_are_not_sent`. Regression module: `kill_keeps_calls`.
- **F3 (fixed): a negative amount lowered the counters**. The value and
  `max_payment` sets held only positive amounts, so the model never tried a negative one; the code counted it,
  which lowered the spend and lifted the caps. The sets now include a negative amount and zero. **Fix:** the policy
  refuses a negative amount (rule `negative-amount`) and a non-integer one (`bad-amount`) before counting; Connect is
  modelled as refusing a negative transfer, so the violation shows as real spend past the cap, and `inv1_caps` also
  requires the counters to stay non-negative. Tests: `test_negative_and_junk_amounts_are_refused_never_counted`,
  `test_negative_zero_hex_and_junk_amounts` and the property test. Regression module: `negative_counted`.
- **F4 (fixed): a call queued behind a slow send was sent after the kill**. The
  broker checked for a kill before taking the upstream lock, and the model declined every waiting call at the kill
  at once, which hid it. The model now has a queued state between approval and send. **Fix:** the broker checks
  again under the lock, right before the send, for MCP tool calls and the REST value routes alike (one helper; the
  model's calls stand for both); a call already being sent at the kill is named in the timed-out
  command's output, with its request id. Test: `test_a_call_queued_behind_a_slow_send_is_not_sent_after_the_kill`.
  Regression module: `kill_misses_queued`.
- **F5 (fixed): the gate's pending floor was lost on a restart**. The model kept
  the floor across `crashAndResume`; the code kept it in memory only, so after `--resume` the facilitator could sign
  the slot Connect's pending request held. **Fix:** the floor is saved in the session state when it is set and
  restored on `--resume`; the model saves it in `savedFloor`. The model saves atomically with the send; the code
  saves right after the call returns, so a crash in that window still loses it. Test:
  `test_the_gates_pending_floor_survives_a_restart`. Regression module: `floor_lost_on_crash`.
- **F6 (fixed): a mech request without `priority_mech` was priced in the chain's default asset**, so a mech paid
  in a stablecoin was under-counted. **Fix:** every `mech_request` must name its mech, whose own asset prices
  `max_payment`. Regression module: `default_asset`.
- **F7 (fixed): a paid model call whose follow-up failed wasn't counted.** **Fix:** a request that went out is
  counted even when its follow-up fails (`MechModel.query`). Regression module: `paid_uncounted`.
- **F8 (fixed): the shell timed out a command while one of its broker calls waited for approval**, so the agent saw
  a timeout for a call that could still be sent. **Fix:** the shell holds its deadline while the broker handles a
  call of the command. Regression module: `no_hold`.
- **F9 (fixed): an uncapped hold let broker calls keep a command alive** indefinitely. **Fix:** the hold is capped
  at the broker call's deadline. Regression module: `hold_uncapped`.
- **A1 (open assumption): the gate's floor drop is safe only if the request has expired.** The gate drops the floor
  after 2 failed 300 s waits, assuming the agent's Connect request can no longer settle. If it can (Connect's
  off-chain request lives longer than about 10 minutes), the facilitator signs the same slot and pauses. To check
  with the Connect team (the request's validity window); with one payer (`one_payer`) the question goes away.
- **Model fixes found while writing it** (not design bugs): the facilitator never signs below the on-chain nonce;
  the shell's deadline fires on time (time doesn't pass a due deadline).
