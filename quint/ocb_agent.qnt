// ocb_agent — the broker's session policy, spend across crashes and --resume, the marketplace nonce shared by the
// facilitator and Connect (MarketplaceGate), the deadline of a broker call against the shell's, and the turn's step
// limit (see README.md).
//
// Abstractions (see README):
// - Money is in US cents. Caps are scaled down so random runs hit them: native $0.10 (10), mech 40 cents, 4 value
//   calls (20 in the code), a model budget of 12 cents ($0.05 in the code).
// - Two mechs: mA is paid in the native token (1 cent per base unit), mB in a USD stablecoin (10 cents per base
//   unit). The chain's default asset (the fallback that `priority_mech` replaced) is the native one.
// - Connect is idempotent per request id: a request id is paid at most once, whatever its arguments.
// - The marketplace nonce is one integer per safe; facilitator rows and Connect requests each take a slot and settle
//   in nonce order. A settlement whose slot was already used pauses the facilitator (as happened in live use).
// - Time is an integer tick for the deadline part: the shell deadline is 2 ticks, the broker call deadline 4.
//
// Switches (module parameters), `main` = the shipped design:
//   REQUIRE_PRIORITY  true = every mech_request names its mech;  false = the default-asset fallback (an earlier
//                     design: a request without priority_mech priced in the chain's default asset)
//   CONNECT_FALLBACK  false = Connect pays only the named mech;  true = Connect may fall back to another mech (the
//                     residual assumption in README.md)
//   VALUE_GUARD       true = a value tool is never read-only, even if Connect flags it;  false = trust the flag
//   SAVE_ON_SPEND     true = spend is saved the moment it is counted;  false = saved after each command and model
//                     reply only (the earlier code)
//   PAID_COUNTED      true = a paid model call counts even if its follow-up fails;  false = it isn't counted
//   GATE              true = MarketplaceGate on;  false = no gate (the facilitator and Connect collide)
//   ONE_PAYER         true = target design: every request goes through Connect, one nonce allocator, no facilitator
//   HOLD              true = the shell doesn't time out while the broker holds one of its calls
//   HOLD_CAP          true = the hold is capped at the call deadline;  false = uncapped
//   KILL_CANCELS      true = a killed command's waiting broker calls are declined;  false = they stay (earlier code)
//   RECHECK_AT_SEND   true = a call queued behind the upstream lock is checked for a kill again right before it is
//                     sent;  false = it is sent anyway (earlier code)
//   SAVE_FLOOR        true = the gate's pending floor is saved in the session state and restored on --resume;
//                     false = it is in memory only and lost on a crash (earlier code)
//   EXPIRED_BEFORE_DROP true = assumption: a Connect request the gate gave up on (2 failed waits) has expired and
//                     can't settle any more;  false = it may still settle (assumption A1: the facilitator pauses)
//   REFUSE_NEGATIVE   true = a negative value or max_payment is refused (rule 'negative-amount');  false = it is
//                     counted, lowering the counters
module ocb_agent {
  const REQUIRE_PRIORITY: bool
  const CONNECT_FALLBACK: bool
  const VALUE_GUARD: bool
  const SAVE_ON_SPEND: bool
  const PAID_COUNTED: bool
  const GATE: bool
  const ONE_PAYER: bool
  const HOLD: bool
  const HOLD_CAP: bool
  const KILL_CANCELS: bool
  const EXPIRED_BEFORE_DROP: bool
  const REFUSE_NEGATIVE: bool
  const RECHECK_AT_SEND: bool
  const SAVE_FLOOR: bool

  // ---------------------------------------------------------------- constants
  pure val MAXN = 10            // native cap, cents
  pure val MAXM = 40            // mech cap, cents (below 4 calls × 20, so the cap, not the call count, binds)
  pure val MAXC = 4             // value calls
  pure val BUDGET = 12          // model budget, cents
  pure val NONE = "none"
  pure val MECHS = Set("mA", "mB")
  pure def rate(m: str): int = if (m == "mB") 10 else 1           // cents per base unit of the mech's asset
  pure def price(m: str): int = if (m == "mB") 2 else 5           // a request's price in the mech's base units
  pure val DEFAULT_RATE = 1                                         // the chain's default asset (native)
  pure val RIDS = Set(1, 2, 3)
  pure val ARGS = Set(1, 2)
  pure val PAYS = Set(-20, 0, 2, 5, 20)                            // max_payment choices, base units (with a negative and zero)
  pure val VALUES = Set(-8, 0, 1, 4, 8)                             // native value choices, cents (with a negative and zero)
  pure val TOOLS = Set("wallet_info", "safe_transaction", "mech_request")
  pure val VALUE_TOOLS = Set("safe_transaction", "mech_request")
  pure val MODEL_PRICE = 5                                          // a model call, cents
  pure val T_SHELL = 2
  pure val D_CALL = 4
  pure val STEP_LIMIT = 6
  pure val MAX_NUDGES = 3

  // ---------------------------------------------------------------- state
  type Spend = { native: int, mech: int, calls: int, model: int, deposited: int }
  type Policy = {
    mode: str,
    mem: Spend,                    // this process's counters
    saved: Spend,                  // the last state event in the session file
    savedFloor: int,               // the gate's pending floor in that state event, -1 = none
    approved: Set[(int, int, str)],// (request id, args hash, tool)
    paid: Set[int],                // request ids Connect has paid (idempotency)
    flagged: Set[str],             // tools Connect flags read-only
    realNative: int, realMech: int, realModel: int,   // what actually left the safe, whole session, cents
    flows: List[{ src: str, dst: str, amt: int }],
    lastCounters: Spend,           // counters before the latest action (for the retry invariant)
    lastWasReplay: bool,
  }
  type Market = {
    nonce: int,                    // on-chain mapNonces[safe]
    facNext: int,                  // next slot the facilitator signs
    facRows: Set[int],             // facilitator rows held (signed, not settled)
    conn: Set[int],                // Connect requests pending
    paused: bool,
    floor: int,                    // gate's pending floor, -1 = none
    failures: int,                 // consecutive failed waits for the floor
    alloc: int,                    // ONE_PAYER: Connect's sequential allocator
    givenUp: Set[int],             // Connect slots whose floor the gate dropped
  }
  // state: waiting (approval, gate) | queued (approved, waiting on the upstream lock) | sent | declined
  type Call = { arrive: int, state: str, sentAt: int }
  type Turn = {
    now: int,
    cmdStart: int,                 // -1 = no command running
    cmdKilled: bool,               // the shell reported this command timed out
    calls: List[Call],
    lateSend: bool,                // a call was sent after its command was reported timed out
    maxCmdAge: int,                // longest a command has run
    steps: int, nudges: int, turnOver: bool,
  }

  var p: Policy
  var m: Market
  var t: Turn

  pure val ZERO = { native: 0, mech: 0, calls: 0, model: 0, deposited: 0 }

  action init = all {
    nondet mode = Set("ask", "auto", "deny").oneOf()
    nondet mislabel = Set(Set("wallet_info"), Set("wallet_info", "safe_transaction")).oneOf()
    p' = { mode: mode, mem: ZERO, saved: ZERO, savedFloor: -1, approved: Set(), paid: Set(), flagged: mislabel,
           realNative: 0, realMech: 0, realModel: 0, flows: [], lastCounters: ZERO, lastWasReplay: false },
    m' = { nonce: 0, facNext: 0, facRows: Set(), conn: Set(), paused: false, floor: -1, failures: 0, alloc: 0,
           givenUp: Set() },
    t' = { now: 0, cmdStart: -1, cmdKilled: false, calls: [], lateSend: false, maxCmdAge: 0,
           steps: 0, nudges: 0, turnOver: false },
  }

  action keepM = m' = m
  action keepT = t' = t
  action keepP = p' = p

  // ================================================================ policy (ocb_agent/policy.py)
  pure def readOnly(pp: Policy, tool: str): bool =
    pp.flagged.contains(tool) and (not(VALUE_GUARD) or not(VALUE_TOOLS.contains(tool)))

  // a payment out of the safe; a zero-value call moves nothing, so it is no flow
  pure def flow(fs: List[{ src: str, dst: str, amt: int }], dst: str, amt: int): List[{ src: str, dst: str, amt: int }] =
    if (amt > 0) fs.append({ src: "safe", dst: dst, amt: amt }) else fs

  pure def save(pp: Policy): Policy = pp.with("saved", pp.mem)
  pure def saveIfOnSpend(pp: Policy): Policy = if (SAVE_ON_SPEND) save(pp) else pp

  // One broker call. `mech` = NONE when no priority_mech is given; `amt` = max_payment or value (base units).
  // `yes` = the user's answer in ask mode. Returns the new policy.
  pure def call(pp: Policy, tool: str, rid: int, args: int, mech: str, amt: int, yes: bool, actual: str): Policy =
    val before = pp.with("lastCounters", pp.mem).with("lastWasReplay", false)
    if (readOnly(pp, tool))
      // passes unchecked; if a value tool was mislabelled and trusted, it really pays
      if (tool == "safe_transaction" and not(before.paid.contains(rid)) and amt >= 0)
        before.with("realNative", before.realNative + amt).with("paid", before.paid.union(Set(rid)))
              .with("flows", flow(before.flows, "target", amt))
      else if (tool == "mech_request" and not(before.paid.contains(rid)))
        before.with("realMech", before.realMech + price(actual) * rate(actual)).with("paid", before.paid.union(Set(rid)))
              .with("flows", before.flows.append({ src: "safe", dst: actual, amt: price(actual) * rate(actual) }))
      else before
    else if (before.approved.contains((rid, args, tool)))
      before.with("lastWasReplay", true)          // an approved retry: no new check, Connect won't pay twice
    else if (before.mode == "deny") before
    else if (tool == "mech_request" and REQUIRE_PRIORITY and mech == NONE) before
    else if (REFUSE_NEGATIVE and amt < 0) before                    // rule 'negative-amount'
    else
      val usd = if (tool == "safe_transaction") amt
                else amt * (if (mech == NONE) DEFAULT_RATE else rate(mech))
      val isMech = tool == "mech_request"
      val ok = and {
        before.mem.calls + 1 <= MAXC,
        if (isMech) before.mem.mech + usd <= MAXM else before.mem.native + usd <= MAXN,
        before.mode == "auto" or yes,
      }
      if (not(ok)) before
      else
        val counted = before.with("mem", {
            ...before.mem,
            calls: before.mem.calls + 1,
            mech: before.mem.mech + (if (isMech) usd else 0),
            native: before.mem.native + (if (isMech) 0 else usd),
          }).with("approved", before.approved.union(Set((rid, args, tool))))
        // Connect: pays at most once per request id; a mech request pays the mech it routes to, up to max_payment
        val payee = if (mech == NONE or CONNECT_FALLBACK) actual else mech
        val paidNow =
          if (counted.paid.contains(rid)) counted
          else if (not(isMech) and amt < 0) counted                  // Connect refuses a negative transfer
          else if (not(isMech))
            counted.with("realNative", counted.realNative + amt).with("paid", counted.paid.union(Set(rid)))
                   .with("flows", flow(counted.flows, "target", amt))
          else if (price(payee) > amt) counted                       // Connect refuses: price above max_payment
          else counted.with("realMech", counted.realMech + price(payee) * rate(payee))
                      .with("paid", counted.paid.union(Set(rid)))
                      .with("flows", counted.flows.append({ src: "safe", dst: payee, amt: price(payee) * rate(payee) }))
        saveIfOnSpend(paidNow)

  action brokerCall = all {
    nondet tool = TOOLS.oneOf()
    nondet rid = RIDS.oneOf()
    nondet args = ARGS.oneOf()
    nondet mech = MECHS.union(Set(NONE)).oneOf()
    nondet amt = (if (tool == "safe_transaction") VALUES else PAYS).oneOf()
    nondet yes = Set(true, false).oneOf()
    nondet actual = MECHS.oneOf()
    p' = call(p, tool, rid, args, mech, amt, yes, actual),
    keepM, keepT,
  }

  // a marketplace model call (MechModel): the request goes out and is paid; the follow-up may fail
  action modelCall = all {
    p.mem.model + MODEL_PRICE <= BUDGET,
    nondet followUpFails = Set(true, false).oneOf()
    val paid = p.with("realModel", p.realModel + MODEL_PRICE)
                .with("flows", p.flows.append({ src: "prepaid", dst: "modelMech", amt: MODEL_PRICE }))
                .with("lastCounters", p.mem).with("lastWasReplay", false)
    val counted = if (followUpFails and not(PAID_COUNTED)) paid
                  else paid.with("mem", { ...paid.mem, model: paid.mem.model + MODEL_PRICE })
    // the agent saves state after a reply returns; a failed follow-up raises before that save
    p' = if (followUpFails) saveIfOnSpend(counted) else save(counted),
    keepM, keepT,
  }

  // the facilitator pre-deposit: bypasses the broker, bounded by its allowance (here 6)
  action deposit = {
    nondet amt = Set(2, 3).oneOf()
    all {
      p.mem.deposited + amt <= 6,
      p' = saveIfOnSpend(p.with("mem", { ...p.mem, deposited: p.mem.deposited + amt })
                           .with("flows", p.flows.append({ src: "safe", dst: "balanceTracker", amt: amt }))
                           .with("lastCounters", p.mem).with("lastWasReplay", false)),
      keepM, keepT,
    }
  }

  // the state event after each command (the gate's current floor with it)
  action commandEnds = all { p' = save(p).with("savedFloor", m.floor), keepM, keepT }

  // the process dies; `--resume` restores the last saved state (the gate's failed-wait count isn't saved)
  action crashAndResume = all {
    p' = p.with("mem", p.saved).with("lastCounters", p.saved).with("lastWasReplay", false),
    m' = m.with("floor", if (SAVE_FLOOR) p.savedFloor else -1).with("failures", 0),
    keepT,
  }

  // ================================================================ marketplace nonce (marketplace_gate.py)
  pure def gateOpenForModel(mm: Market): bool = not(GATE) or mm.floor < 0 or mm.nonce >= mm.floor

  action facSign = all {
    not(ONE_PAYER), not(m.paused), gateOpenForModel(m), m.facRows.size() < 3,
    // the facilitator signs its next slot, never below the on-chain nonce (it reads mapNonces)
    val slot = if (m.facNext > m.nonce) m.facNext else m.nonce
    m' = m.with("facRows", m.facRows.union(Set(slot))).with("facNext", slot + 1)
          .with("floor", if (GATE and m.floor >= 0 and m.nonce >= m.floor) -1 else m.floor)
          .with("failures", 0),
    keepP, keepT,
  }

  // the gate's wait for the floor times out (the agent's request never settles); after 2, the floor is dropped
  action gateWaitFails = all {
    GATE, not(ONE_PAYER), m.floor >= 0, m.nonce < m.floor,
    m' = if (m.failures + 1 >= 2) m.with("floor", -1).with("failures", 0).with("givenUp", m.givenUp.union(Set(m.floor - 1)))
         else m.with("failures", m.failures + 1),
    keepP, keepT,
  }

  // facilitator rows settle in order; a row whose slot was already used pauses the facilitator
  action facSettle = all {
    m.facRows != Set(),
    val r = m.facRows.fold(1000, (a, x) => if (x < a) x else a)
    all {
      r <= m.nonce,
      m' = if (r == m.nonce) m.with("nonce", m.nonce + 1).with("facRows", m.facRows.exclude(Set(r)))
           else m.with("paused", true).with("facRows", m.facRows.exclude(Set(r))),
    },
    keepP, keepT,
  }

  // the agent's own mech_request through Connect: signed at the on-chain nonce (ONE_PAYER: Connect's own allocator)
  action connSign = all {
    m.conn.size() < 2,
    if (ONE_PAYER) all {
      val slot = if (m.alloc > m.nonce) m.alloc else m.nonce
      m' = m.with("conn", m.conn.union(Set(slot))).with("alloc", slot + 1),
    } else all {
      not(GATE) or (m.facRows == Set() and (m.floor < 0 or m.nonce >= m.floor)),
      m' = m.with("conn", m.conn.union(Set(m.nonce)))
            .with("floor", if (GATE) m.nonce + 1 else -1).with("failures", 0),
    },
    // the new floor is saved at once (MarketplaceGate.on_floor)
    p' = if (not(ONE_PAYER) and GATE) p.with("savedFloor", m.nonce + 1) else p,
    keepT,
  }

  // ONE_PAYER: model calls also go through Connect's allocator
  action onePayerModelSign = all {
    ONE_PAYER, m.conn.size() < 2,
    val slot = if (m.alloc > m.nonce) m.alloc else m.nonce
    m' = m.with("conn", m.conn.union(Set(slot))).with("alloc", slot + 1),
    keepP, keepT,
  }

  action connSettle = all {
    m.conn != Set(),
    nondet s = m.conn.oneOf()
    all {
      s <= m.nonce,
      not(EXPIRED_BEFORE_DROP and m.givenUp.contains(s)),
      m' = if (s == m.nonce) m.with("nonce", m.nonce + 1).with("conn", m.conn.exclude(Set(s)))
           else m.with("conn", m.conn.exclude(Set(s))),        // its slot was used: rejected, nothing paid
    },
    keepP, keepT,
  }

  action connExpire = all {
    m.conn != Set(),
    nondet s = m.conn.oneOf()
    m' = m.with("conn", m.conn.exclude(Set(s))),
    keepP, keepT,
  }

  // ================================================================ deadlines (shell.py, broker.py, cli.py)
  pure def waiting(tt: Turn): bool = tt.calls.select(c => c.state == "waiting").length() > 0
  // a call the broker is still handling (Broker.busy): waiting, or queued behind the upstream lock
  pure def pending(tt: Turn): bool = tt.calls.select(c => c.state == "waiting" or c.state == "queued").length() > 0

  action startCommand = all {
    t.cmdStart < 0, not(t.turnOver),
    t' = t.with("cmdStart", t.now).with("cmdKilled", false).with("calls", []),
    keepP, keepM,
  }

  action arriveCall = all {
    t.cmdStart >= 0, t.calls.length() < 3,
    t' = t.with("calls", t.calls.append({ arrive: t.now, state: "waiting", sentAt: -1 })),
    keepP, keepM,
  }

  // the user answers yes (or the gate opens): the call queues behind the upstream lock, where another call may be
  // slow to send
  action approve = all {
    t.calls.indices().exists(i => t.calls[i].state == "waiting"),
    nondet i = t.calls.indices().filter(i => t.calls[i].state == "waiting").oneOf()
    t' = t.with("calls", t.calls.replaceAt(i, { ...t.calls[i], state: "queued" })),
    keepP, keepM,
  }

  // the call takes the lock and is sent; with the re-check, a call whose command was killed is declined instead
  action send = all {
    t.calls.indices().exists(i => t.calls[i].state == "queued"),
    nondet i = t.calls.indices().filter(i => t.calls[i].state == "queued").oneOf()
    if (RECHECK_AT_SEND and t.cmdKilled)
      t' = t.with("calls", t.calls.replaceAt(i, { ...t.calls[i], state: "declined" }))
    else
      t' = t.with("calls", t.calls.replaceAt(i, { ...t.calls[i], state: "sent", sentAt: t.now }))
            .with("lateSend", t.lateSend or t.cmdKilled),
    keepP, keepM,
  }

  pure def held(tt: Turn): bool =
    HOLD and pending(tt) and (not(HOLD_CAP) or tt.now < tt.cmdStart + T_SHELL + D_CALL)
  // the shell's deadline fires on time: time doesn't pass a due, unheld deadline
  pure def deadlineDue(tt: Turn): bool = tt.cmdStart >= 0 and tt.now >= tt.cmdStart + T_SHELL and not(held(tt))

  action tick = all {
    t.now < 30, not(deadlineDue(t)),
    val n = t.now + 1
    val calls = t.calls.foldl([], (acc, c) =>
      acc.append(if (c.state == "waiting" and n >= c.arrive + D_CALL) { ...c, state: "declined" } else c))
    val age = if (t.cmdStart >= 0) n - t.cmdStart else 0
    t' = t.with("now", n).with("calls", calls).with("maxCmdAge", if (age > t.maxCmdAge) age else t.maxCmdAge),
    keepP, keepM,
  }

  // the shell's deadline check: the kill declines the waiting calls at once; a queued call is past that check and
  // learns of the kill only at the send (RECHECK_AT_SEND)
  action shellDeadline = all {
    deadlineDue(t),
    all {
      val calls = if (KILL_CANCELS)
                    t.calls.foldl([], (acc, c) => acc.append(if (c.state == "waiting") { ...c, state: "declined" } else c))
                  else t.calls
      t' = t.with("cmdStart", -1).with("cmdKilled", true).with("calls", calls),
    },
    keepP, keepM,
  }

  action commandFinishes = all {
    t.cmdStart >= 0, not(pending(t)),
    t' = t.with("cmdStart", -1).with("cmdKilled", false)
          .with("steps", t.steps + 1).with("turnOver", t.steps + 1 >= STEP_LIMIT),
    keepP, keepM,
  }

  // a model reply without an action: nudged up to MAX_NUDGES times (unattended), then the turn ends
  action replyWithoutAction = all {
    t.cmdStart < 0, not(t.turnOver),
    t' = t.with("nudges", t.nudges + 1).with("steps", t.steps + 1)
          .with("turnOver", t.nudges + 1 >= MAX_NUDGES or t.steps + 1 >= STEP_LIMIT),
    keepP, keepM,
  }

  action step = any {
    brokerCall, modelCall, deposit, commandEnds, crashAndResume,
    facSign, gateWaitFails, facSettle, connSign, onePayerModelSign, connSettle, connExpire,
    startCommand, arriveCall, approve, send, tick, shellDeadline, commandFinishes, replyWithoutAction,
  }

  // ================================================================ invariants (README.md)
  // 1. caps: counted and real spend within the caps, for the whole session (3b: across crashes)
  val inv1_caps = and {
    p.mem.native <= MAXN, p.mem.mech <= MAXM, p.mem.calls <= MAXC,
    p.mem.native >= 0, p.mem.mech >= 0,                            // a negative amount never lowers a counter
    p.realNative <= MAXN, p.realMech <= MAXM,
  }
  // 2. an approved (id, args) replayed changes no counter
  val inv2_retry = p.lastWasReplay implies p.mem == p.lastCounters
  // 3. model budget and deposit allowance, whole session
  val inv3_model = p.realModel <= BUDGET and p.mem.deposited <= 6
  // 4. with the gate (or ONE_PAYER), the facilitator is never paused by a collision
  val inv4_noCollision = not(m.paused)
  // 6. bounded turns and waits; no value call sent after its command was reported timed out
  val inv6_turn = t.steps <= STEP_LIMIT and t.nudges <= MAX_NUDGES
  val inv6_noLateSend = not(t.lateSend)
  val inv6_cmdBounded = t.maxCmdAge <= T_SHELL + D_CALL + 1
  // 8. every flow has one source and one destination, both known
  pure val SOURCES = Set("safe", "prepaid")
  pure val SINKS = Set("target", "mA", "mB", "modelMech", "balanceTracker")
  val inv8_flows = p.flows.foldl(true, (ok, f) => ok and SOURCES.contains(f.src) and SINKS.contains(f.dst) and f.amt > 0)

  val allInvariants = and {
    inv1_caps, inv2_retry, inv3_model, inv4_noCollision, inv6_turn, inv6_noLateSend, inv6_cmdBounded,
    inv8_flows,
  }

  // ================================================================ witnesses (expected to be violated: reachable)
  val wNeverMechCapReached = p.mem.mech + 2 * 10 <= MAXM
  val wNeverReplay = not(p.lastWasReplay)
  val wNeverResumedWithSpend = not(p.saved.mech > 0 and p.mem == p.saved and p.realMech > 0)
  val wNeverModelBudgetUsed = p.realModel < 10
  val wNeverConnSettled = m.nonce < 2
  val wNeverPaused = not(m.paused)
  val wNeverHeldSend = t.calls.select(c => c.state == "sent" and c.sentAt >= c.arrive + T_SHELL).length() == 0
  val wNeverTurnOver = not(t.turnOver)
}

// ==================================================================== instances
// The shipped design (code after the formal-checking fixes) plus the one assumption it needs.
module main {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}

// The target design, one payer: every request through Connect's allocator, no facilitator, no gate. No pause
// reachable.
module one_payer {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = false, ONE_PAYER = true, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = false,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}

// ---- regressions and assumptions: each flips one switch of `main` and must violate the named invariant

// formal checking F1: spend saved only after each command (the earlier code): a crash mid-command loses it (inv1)
module save_after_command {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = false,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// formal checking F2: a killed command's waiting calls stay and can be sent late (the earlier code) (inv6_noLateSend)
module kill_keeps_calls {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = false,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = false, SAVE_FLOOR = true).*
}
// finding F4: a call queued behind a slow send passed the kill check before the kill, and is sent after it
// (inv6_noLateSend)
module kill_misses_queued {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = false, SAVE_FLOOR = true).*
}
// finding F5: the gate's pending floor in memory only; after a crash the facilitator signs the slot Connect's pending
// request holds (inv4_noCollision)
module floor_lost_on_crash {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = false).*
}
// A1 (assumption): a Connect request may settle after the gate gave up on it: the facilitator pauses (inv4)
module late_settle_after_drop {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = false,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// no gate: the facilitator and Connect sign the same nonce, and the facilitator pauses, as seen in live use (inv4)
module no_gate {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = false, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// finding F6: the default-asset fallback under-counts a stablecoin mech (inv1)
module default_asset {
  import ocb_agent(REQUIRE_PRIORITY = false, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// residual assumption: Connect falls back from the named mech to one paid in another asset (inv1)
module connect_fallback {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = true, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// the value guard off: a value tool Connect mislabels read-only is trusted and pays uncounted (inv1)
module no_value_guard {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = false, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// finding F7: a paid model call whose follow-up fails isn't counted (inv3)
module paid_uncounted {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = false, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// finding F8: no hold, the shell times out under a pending approval (inv6_noLateSend)
module no_hold {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = false, HOLD_CAP = true, KILL_CANCELS = false,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = false, SAVE_FLOOR = true).*
}
// finding F9: an uncapped hold lets broker calls keep a command alive (inv6_cmdBounded)
module hold_uncapped {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = false, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = true, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
// finding F3: a negative value or max_payment counted lowers the counters and lifts the caps (inv1)
module negative_counted {
  import ocb_agent(REQUIRE_PRIORITY = true, CONNECT_FALLBACK = false, VALUE_GUARD = true, SAVE_ON_SPEND = true,
               PAID_COUNTED = true, GATE = true, ONE_PAYER = false, HOLD = true, HOLD_CAP = true, KILL_CANCELS = true,
               EXPIRED_BEFORE_DROP = true,
               REFUSE_NEGATIVE = false, RECHECK_AT_SEND = true, SAVE_FLOOR = true).*
}
