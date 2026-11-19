"""Chain data for the Olas EVM chains. Data only: olas-ocb-agent's logic works on any EVM chain.

- **Olas EVM chains** (2026-11-09): Ethereum, Gnosis, Polygon, Arbitrum, Optimism, Base and Celo (the
  ServiceManager upgrade of June 2026), plus Mode (still configured by Pearl and Connect) and Robinhood
  Chain (deployment approved in Q3 2026).
- **Any other EVM chain** works once the user supplies its native token's USD price
  (`--native-usd <chain>=<price>`); without one, value calls there are refused with that instruction.
- **Mech payment assets** come from the mech itself (Connect's `mech_tools` reports `payment_token`), so
  every mech request names its mech. The facilitator's own assets are interim (`interim/facilitator_assets.py`).
- **Prices are rough** and only make USD session caps comparable across chains. They're a stopgap,
  to be replaced by caps that Connect enforces itself on scoped per-session tokens. A supported-chains list
  published by Connect would replace this table.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext

NATIVE = "native"
STABLE6 = "stable6"  # a 6-decimal USD stablecoin (USDC on most chains, USDG on Robinhood)


@dataclass(frozen=True)
class ChainInfo:
    native_symbol: str
    native_usd: float  # rough


OLAS_EVM_CHAINS: dict[str, ChainInfo] = {
    "ethereum": ChainInfo("ETH", 3000.0),
    "gnosis": ChainInfo("xDAI", 1.0),
    "polygon": ChainInfo("POL", 0.25),
    "arbitrum": ChainInfo("ETH", 3000.0),
    "optimism": ChainInfo("ETH", 3000.0),
    "base": ChainInfo("ETH", 3000.0),
    "celo": ChainInfo("CELO", 0.5),
    "mode": ChainInfo("ETH", 3000.0),
    "robinhood": ChainInfo("ETH", 3000.0),
}

# Tokens counted at $1 when a mech is paid in them (matched on the on-chain `symbol()`).
STABLE_SYMBOLS = {"USDC", "USDC.E", "USDBC", "USDT", "USDT0", "DAI", "XDAI", "WXDAI", "USDG", "USDS"}


def native_usd(chain: str | None, overrides: dict[str, float] | None = None) -> float | None:
    """The native token's USD price: the user's override first, then the table; None if unknown."""
    if chain and overrides and chain in overrides:
        return overrides[chain]
    info = OLAS_EVM_CHAINS.get(chain or "")
    return info.native_usd if info else None


def native_symbol(chain: str | None) -> str:
    """The native token's symbol (xDAI on gnosis); 'native' on a chain outside the table."""
    info = OLAS_EVM_CHAINS.get(chain or "")
    return info.native_symbol if info else "native"


def fmt_usd(usd: float) -> str:
    """$1,500.00 and $0.50; small amounts keep enough digits to read ($0.0010, <$0.0001); negative ones as -$0.50."""
    sign, usd = ("-", -usd) if usd < 0 else ("", usd)
    if usd >= 0.01 or usd == 0:
        return f"{sign}${usd:,.2f}"
    return f"{sign}${usd:.4f}" if usd >= 0.0001 else f"{sign}<$0.0001"


def fmt_amount(n: int, decimals: int, symbol: str, usd_price: float | None = None) -> str:
    """Base units as a person reads them: 500000000000000000, 18, 'xDAI', 1.0 -> '0.5 xDAI (about $0.50)'."""
    with localcontext() as ctx:
        ctx.prec = 100  # exact for any uint256 (78 digits)
        q = Decimal(n).scaleb(-decimals)
        out = f"{format(q.normalize(), 'f') if n else '0'} {symbol}"
    return out if usd_price is None else f"{out} (about {fmt_usd(float(q) * usd_price)})"


def parse_prices(pairs: list[str] | None) -> dict[str, float]:
    """['arbitrum=3100', 'mychain=0.2'] -> {'arbitrum': 3100.0, 'mychain': 0.2}."""
    out = {}
    for p in pairs or []:
        chain, sep, price = p.partition("=")
        try:
            if not sep or not chain:
                raise ValueError
            out[chain.strip()] = float(price)
        except ValueError:
            raise ValueError(f"--native-usd: expected <chain>=<USD price>, like gnosis=1.0; got {p!r}") from None
    return out
