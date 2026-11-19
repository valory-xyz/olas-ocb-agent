"""The asset the mech facilitator charges in, per chain it serves.

Interim: goes away once the facilitator's mechs are full marketplace mechs. Their payment asset then
comes from the mech itself (`payment_token`), like every other mech's.
"""

from __future__ import annotations

from ..chains import NATIVE, STABLE6, native_symbol, native_usd

# NATIVE or STABLE6 per chain the facilitator serves (2026-11-08: Gnosis in xDAI; Polygon, Optimism and Base in USDC)
FACILITATOR_ASSET = {"gnosis": NATIVE, "polygon": STABLE6, "optimism": STABLE6, "base": STABLE6}


def facilitator_asset(chain: str | None, overrides: dict[str, float] | None = None) -> tuple[int, float] | None:
    """(decimals, USD price) of the asset the mech facilitator charges on this chain; None if it doesn't serve it."""
    kind = FACILITATOR_ASSET.get(chain or "")
    if kind is None:
        return None
    if kind == NATIVE:
        return 18, native_usd(chain, overrides)
    return 6, 1.0


def facilitator_symbol(chain: str | None) -> str:
    """The symbol of that asset: the native token's (xDAI on gnosis), or USDC."""
    return native_symbol(chain) if FACILITATOR_ASSET.get(chain or "") == NATIVE else "USDC"
