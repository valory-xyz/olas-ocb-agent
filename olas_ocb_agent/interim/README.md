# Interim code

Everything in this package is a stopgap: each module goes away once Pearl Connect or the Olas mech marketplace
does its job itself.

| Module | Goes away once | Replaced by | Used from |
|---|---|---|---|
| `facilitator.py` (`FacilitatorModel`) | chat models are served as full marketplace mechs that Connect pays for | `MechModel` (`model.py`) | `runtime.make_model` (`facilitator:` specs), the CLI's default model |
| `marketplace_gate.py` (`MarketplaceGate`) | every off-chain request, model calls included, goes through Connect, so Connect is the only payer per Safe | nothing: one payer can't collide with itself | `runtime.build`, when the model is a `FacilitatorModel` |
| `facilitator_assets.py` | the facilitator's mechs are full marketplace mechs, priced in their own payment asset | the mech's `payment_token` (`runtime.mech_asset_lookup`) | `runtime.build` (the model budget's asset) |
| `readonly_fallback.py` (`FALLBACK_READ_ONLY`) | Connect flags its read-only tools (MCP's `readOnlyHint`) | Connect's flags (`policy.read_only_from`) | `policy.Policy` |

Removing a module: delete it, delete its import, and delete the tests named after it (`tests/test_facilitator*.py`
for the facilitator and the gate).
