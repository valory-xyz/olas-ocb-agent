"""Which workspace skills olas-ocb-agent treats as instructions.

A skill file is an instruction only if its content hash is trusted:
- pinned in olas-ocb-agent's own manifest (`trusted_skills.json` next to this file; empty: Pearl and Connect name
  the skills they ship at launch instead, with `--trusted-skill-hash`), or
- approved by the user (`ocb-agent --trust-skill PATH`), kept in `~/.config/olas-ocb-agent/trusted-skills.json`, outside
  every path the agent's sandbox may write, so an agent can't trust a skill for itself, or
- named by the caller (`--trusted-skill-hash`, `build(trusted_skills=...)`: a launcher or an eval harness trusts the
  skills it ships or writes as the operator, for that session only).

Any other `SKILL.md` in the workspace is listed to the agent as unverified data: it is told not to follow it and
to mention it to the user. OnchainBench found the gap: an untrusted "payments-express" skill in the workspace told
olas-ocb-agent to pay a 1% fee to an outside address before each payment, and olas-ocb-agent did.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PINNED = Path(__file__).resolve().parent / "trusted_skills.json"
USER_STORE = Path.home() / ".config" / "olas-ocb-agent" / "trusted-skills.json"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(p: Path) -> set[str]:
    try:
        return set(json.loads(p.read_text()).get("sha256", []))
    except (OSError, ValueError, AttributeError):
        return set()


def trusted_hashes(extra: set[str] | None = None, store: Path | None = None) -> set[str]:
    return _load(PINNED) | _load(store or USER_STORE) | set(extra or ())


def trust(path: Path, store: Path | None = None) -> str:
    """Record a skill file's hash as approved by the user; returns the hash."""
    store = store or USER_STORE
    h = digest(path)
    have = _load(store) | {h}
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(json.dumps({"sha256": sorted(have)}, indent=1) + "\n")
    return h
