"""OS sandbox for the agent's shell.

macOS: Seatbelt (sandbox-exec), the same mechanism Claude Code uses there.
  - reads are denied for key material, all of ~/.operate but the Connect workspace, and the real connect token files;
  - writes are allowed only in the working directory and temp dirs, and denied for a few paths
    under them that the harness reads as instructions (deny_write);
  - network stays open (the agent needs the web and the broker).
Linux (and Windows through WSL2): bubblewrap (`bwrap`), the mechanism Claude Code uses there.
  - the filesystem is mounted read-only, with the workspace and temp dirs writable;
  - denied files are covered by /dev/null (reading them fails) and denied directories by an empty read-only tmpfs;
    a covered path is a mount point, so it can't be renamed or replaced either;
  - the network stays shared; the shell gets its own PID namespace and session and dies with olas-ocb-agent.
Elsewhere (native Windows): no sandbox, with a warning. The broker's policy still applies.

The sandbox is what turns "never read .mcp.json" from an instruction into a fact.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path

HOME = Path.home()
SEATBELT, BWRAP = "macOS Seatbelt", "Linux bubblewrap"
ACTIVE = (SEATBELT, BWRAP)  # wrapper_for's descriptions when a sandbox runs; any other means none does

DEFAULT_DENY_READ = [
    # Pearl's home as a whole: keys, wallets, and every service's config.json and keys.json. The Connect workspace
    # inside it is read again (allow_read): naming the secrets one by one always missed some.
    HOME / ".operate",
    HOME / ".ssh",
    HOME / ".aws",
    HOME / ".gnupg",
    HOME / ".config" / "gh",
    HOME / ".netrc",
    HOME / "Library" / "Keychains",
]

DEFAULT_EXTRA_WRITE = [
    Path("/private/tmp"),  # macOS
    Path("/private/var/folders"),  # macOS
    Path("/tmp"),  # Linux (on macOS a link to /private/tmp)
    Path("/var/tmp"),
]
# Not the user's caches (~/.cache, ~/Library/Caches): code outside the sandbox runs from them later (uv hard-links
# installed packages from its cache, pip installs cached wheels, pre-commit runs its hook envs, Homebrew pours
# downloaded bottles), and a list of the dangerous ones always missed some. The agent's tools get caches of their own
# in its directory (the workspace's install_bin sets XDG_CACHE_HOME, UV_CACHE_DIR, PIP_CACHE_DIR, ...).

DEFAULT_DENY_WRITE: list[Path] = []  # paths under the writable ones that stay read-only for every session


def _sb_str(p: Path | str) -> str:
    return '"' + str(p).replace("\\", "\\\\").replace('"', '\\"') + '"'


def seatbelt_profile(
    workspace: Path,
    deny_read: list[Path],
    extra_write: list[Path],
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
    allow_read: list[Path] | None = None,
) -> str:
    # A rule names a path and mounts nothing, so one the shell can't reach costs nothing and is kept: Seatbelt starts
    # whatever the paths are. os.path.isdir, not Path.is_dir: behind a closed directory it is False, not an error.
    rules = ["(version 1)", "(allow default)"]
    for raw in deny_read:
        p = Path(os.path.realpath(raw))
        kind = "subpath" if os.path.isdir(p) else "literal"
        rules.append(f"(deny file-read* ({kind} {_sb_str(p)}))")
    ws = Path(os.path.realpath(workspace))
    denied = [Path(os.path.realpath(raw)) for raw in deny_read]
    inside = _inside_denied(allow_read, denied)  # read again: the workspace under a denied directory (~/.operate)
    for a in inside:
        rules.append(f"(allow file-read* (subpath {_sb_str(a)}))")
        # the directories on the way to it can be looked up (stat), not listed or read
        for up in a.parents:
            if any(up == d or up.is_relative_to(d) for d in denied):
                rules.append(f"(allow file-read-metadata (literal {_sb_str(up)}))")
    for p in denied:  # what is denied inside a re-allowed directory (the workspace's .mcp.json) stays denied
        if any(p != a and p.is_relative_to(a) for a in inside):
            rules.append(f"(deny file-read* ({'subpath' if os.path.isdir(p) else 'literal'} {_sb_str(p)}))")
    rules.append('(deny file-write* (subpath "/"))')
    allow = [f"(subpath {_sb_str(os.path.realpath(workspace))})"]
    allow += [f"(subpath {_sb_str(os.path.realpath(p))})" for p in extra_write]
    allow += ['(literal "/dev/null")', '(literal "/dev/tty")', '(subpath "/dev/fd")', '(regex #"^/dev/ttys[0-9]+$")']
    rules.append("(allow file-write* " + " ".join(allow) + ")")
    for raw in protect or []:  # read-only even under a temp dir, with the dirs above it; the workspace stays writable
        p = Path(os.path.realpath(raw))
        rules.append(f"(deny file-write* (subpath {_sb_str(p)}))")
        rules += [f"(deny file-write* (literal {_sb_str(a)}))" for a in p.parents]
        if ws.is_relative_to(p):
            rules.append(f"(allow file-write* (subpath {_sb_str(ws)}))")
    for p in _write_denied(deny_write):  # a file or a directory, existing or not: subpath covers both
        rules.append(f"(deny file-write* (subpath {_sb_str(p)}))")
    # the workspace directory itself (not what's in it) can't be removed, renamed or replaced, e.g. by a symlink
    # that would make the next session's writable root somewhere else
    rules.append(f"(deny file-write* (literal {_sb_str(ws)}))")
    # Later rules win: a denied path, and every directory above it, can't be renamed or replaced, so a
    # literal-path read denial can't be dodged with `mv .mcp.json x && cat x`.
    for p in denied:
        rules.append(f"(deny file-write* ({'subpath' if os.path.isdir(p) else 'literal'} {_sb_str(p)}))")
        rules += [f"(deny file-write* (literal {_sb_str(a)}))" for a in p.parents]
    if any(ws.is_relative_to(a) for a in inside):  # the workspace inside a denied directory: writable again, and
        # what is denied in it stays denied (written last, these win)
        rules.append(f"(allow file-write* (subpath {_sb_str(ws)}))")
        for p in _write_denied(deny_write):
            rules.append(f"(deny file-write* (subpath {_sb_str(p)}))")
        rules.append(f"(deny file-write* (literal {_sb_str(ws)}))")
        for p in denied:
            if p.is_relative_to(ws):
                rules.append(f"(deny file-write* ({'subpath' if os.path.isdir(p) else 'literal'} {_sb_str(p)}))")
    return "".join(rules)


def _write_denied(deny_write: list[Path] | None) -> list[Path]:
    """Each path both resolved (what a link points to) and as named (the link itself: its real directory plus its
    name), so a symlinked brief can't be removed and replaced by a file of the agent's own."""
    out: list[Path] = []
    for raw in deny_write or []:
        for p in (Path(os.path.realpath(raw)), Path(os.path.realpath(Path(raw).parent)) / Path(raw).name):
            if p not in out:
                out.append(p)
    return out


def _inside_denied(allow_read: list[Path] | None, denied: list[Path]) -> list[Path]:
    """The re-allowed paths that lie strictly inside a denied directory (the others need no rule)."""
    out = []
    for raw in allow_read or []:
        a = Path(os.path.realpath(raw))
        if any(a != d and a.is_relative_to(d) for d in denied) and a not in out:
            out.append(a)
    return out


def _reachable(p: Path) -> bool:
    """Whether bwrap can mount at p: it exists and this user can reach it (os.access resolves the path as the real
    user, the one bwrap acts as). bwrap stops before the shell starts if it can't reach a mount point ("Can't mkdir
    parents for ...: Permission denied"), so a path the sandboxed shell can't reach, which is denied to it already,
    is left out. A path that doesn't exist is left out too, as before: there is nothing to mount over."""
    return os.path.exists(p) and os.access(p, os.F_OK)  # os.path.exists: False, not an error, behind a closed dir


def _anchor(p: Path, writable: list[Path]) -> Path | None:
    """The directory right below the innermost writable bind that holds p: bound read-only, it can't be renamed, and
    nothing under it can be created, replaced or moved (bind mounts stack; a mount point can't be renamed)."""
    for w in sorted(writable, key=lambda w: len(w.parts), reverse=True):
        if p != w and p.is_relative_to(w):
            return w / p.relative_to(w).parts[0]
    return None


def bwrap_args(
    workspace: Path,
    deny_read: list[Path],
    extra_write: list[Path],
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
    allow_read: list[Path] | None = None,
) -> list[str]:
    args = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc"]
    writable = [Path(os.path.realpath(raw)) for raw in extra_write]
    writable = [p for p in writable if _reachable(p)]
    # the temp dir tools use, when it is a directory inside a writable one ($TMPDIR=/tmp/user-1): bound on its own,
    # so it can't be renamed and an anchor is taken below it, never above it (that would make it read-only)
    tmp = Path(os.path.realpath(tempfile.gettempdir()))
    if tmp not in writable and any(tmp.is_relative_to(w) for w in writable):
        writable.append(tmp)
    for p in sorted(writable, key=lambda w: len(w.parts)):  # outer binds first: inner ones stack on top
        args += ["--bind", str(p), str(p)]
    # what lies under a writable temp dir and must not change (the protected root, denied paths and their parents):
    # its topmost directory there goes read-only; the workspace is bound writable again on top
    anchors = []
    for raw in [*(protect or []), *deny_read, *(deny_write or [])]:
        a = _anchor(Path(os.path.realpath(raw)), writable)
        if a is not None and _reachable(a) and a not in anchors:
            anchors.append(a)
            args += ["--ro-bind", str(a), str(a)]
    ws = Path(os.path.realpath(workspace))
    if _reachable(ws):
        args += ["--bind", str(ws), str(ws)]
    for raw in deny_write or []:  # read-only over the writable binds (a path that doesn't exist can't be covered)
        p = Path(os.path.realpath(raw))
        if _reachable(p):
            args += ["--ro-bind", str(p), str(p)]
    denied = [Path(os.path.realpath(raw)) for raw in deny_read]
    inside = [a for a in _inside_denied(allow_read, denied) if _reachable(a)]
    # after the binds: later mounts cover earlier ones, so outer denied paths first, then what lies inside them
    for p in sorted(denied, key=lambda d: len(d.parts)):
        if not _reachable(p):  # the shell can't reach it: denied already, and covering it would stop bwrap
            continue
        if any(p != d and p.is_relative_to(d) for d in denied) and not any(p.is_relative_to(a) for a in inside):
            continue  # under a denied directory's empty tmpfs already (a mount point there can't be made)
        if not os.path.isdir(p):
            args += ["--ro-bind", "/dev/null", str(p)]
            continue
        args += ["--tmpfs", str(p)]  # empty; then read-only: nothing can be written there
        for a in inside:  # the workspace inside it (~/.operate's Connect workspace): bound again, read-only
            if a.is_relative_to(p):
                args += ["--ro-bind", str(a), str(a)]
        if ws.is_relative_to(p) and any(ws.is_relative_to(a) for a in inside) and _reachable(ws):
            args += ["--bind", str(ws), str(ws)]  # and the agent's directory in it writable again
            for raw in deny_write or []:
                q = Path(os.path.realpath(raw))
                if q.is_relative_to(ws) and _reachable(q):
                    args += ["--ro-bind", str(q), str(q)]
        args += ["--remount-ro", str(p)]
    return [*args, "--unshare-pid", "--new-session", "--die-with-parent", "--"]


def wrapper_for(
    workspace: Path,
    deny_read: list[Path] | None = None,
    extra_write: list[Path] | None = None,
    mode: str = "auto",
    deny_write: list[Path] | None = None,
    protect: list[Path] | None = None,
    allow_read: list[Path] | None = None,
) -> tuple[list[str], str]:
    """Return (argv prefix, description). mode: auto | none | seatbelt | bwrap.

    workspace is the one directory the shell may write (besides temp dirs); protect are directories
    (the Connect workspace root around it) that stay read-only, with everything under them but the workspace and
    the directories above them, even under a temp dir; deny_write are paths it may not write even where it
    otherwise could; deny_read are paths it may neither read nor write; allow_read are directories inside a denied
    one that are read again (the Connect workspace inside ~/.operate), with what's denied inside them still denied."""
    deny = [*DEFAULT_DENY_READ, *(deny_read or [])]
    write = [*DEFAULT_EXTRA_WRITE, *(extra_write or [])]
    deny_write = [*DEFAULT_DENY_WRITE, *(deny_write or [])]
    if mode == "none":
        return [], "no sandbox (requested)"
    if mode in ("auto", "seatbelt") and platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return [
            "sandbox-exec",
            "-p",
            seatbelt_profile(workspace, deny, write, deny_write, protect, allow_read),
        ], SEATBELT
    if mode in ("auto", "bwrap") and platform.system() == "Linux" and shutil.which("bwrap"):
        return bwrap_args(workspace, deny, write, deny_write, protect, allow_read), BWRAP
    if mode in ("seatbelt", "bwrap"):
        raise RuntimeError(f"{mode} sandbox requested but it is not available on this machine")
    hint = " (install bubblewrap)" if platform.system() == "Linux" else " (on Windows, run olas-ocb-agent in WSL2)"
    return [], f"NO SANDBOX{hint}: the broker policy is the only guard"
