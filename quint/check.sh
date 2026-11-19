#!/usr/bin/env bash
# Every Quint check of quint/ocb_agent.qnt: main and one_payer hold all invariants; each witness is reachable in main; each
# regression module violates its invariant. Fixed seed. About 2 minutes.
set -euo pipefail
cd "$(dirname "$0")"
N=${N:-60000}; STEPS=${STEPS:-50}; SEED=${SEED:-11}
quint typecheck ocb_agent.qnt
run() { quint run ocb_agent.qnt --main="$1" --invariant="$2" --max-samples="$N" --max-steps="$STEPS" --seed="$SEED" 2>&1 \
          | grep -E '^\[(ok|violation)\]' | cut -c1-12 || true; }
fail=0
expect() { got=$(run "$1" "$2"); want=$3
  if [[ "$got" == "[$want]"* ]]; then printf '  ok    %-24s %-24s %s\n' "$1" "$2" "$got"
  else printf '  FAIL  %-24s %-24s %s (wanted %s)\n' "$1" "$2" "$got" "$want"; fail=1; fi; }
echo "invariants (must hold)"
expect main allInvariants ok
expect one_payer allInvariants ok
echo "witnesses in main (must be reachable)"
for w in wNeverMechCapReached wNeverReplay wNeverResumedWithSpend wNeverModelBudgetUsed wNeverConnSettled \
         wNeverHeldSend wNeverTurnOver; do expect main "$w" violation; done
echo "regressions and assumptions (must be violated)"
for pair in save_after_command:inv1_caps kill_keeps_calls:inv6_noLateSend kill_misses_queued:inv6_noLateSend \
            floor_lost_on_crash:inv4_noCollision late_settle_after_drop:inv4_noCollision \
            no_gate:inv4_noCollision default_asset:inv1_caps connect_fallback:inv1_caps no_value_guard:inv1_caps \
            paid_uncounted:inv3_model no_hold:inv6_noLateSend hold_uncapped:inv6_cmdBounded \
            negative_counted:inv1_caps; do expect "${pair%%:*}" "${pair##*:}" violation; done
exit $fail
