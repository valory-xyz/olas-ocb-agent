"""A persistent bash session: `cd`, exported variables and activated venvs survive between actions.

mini-swe-agent runs every action in a fresh subprocess, which keeps it simple
but makes chat sessions awkward ("cd into the repo" is forgotten next turn).
Here one bash process lives for the session. Each action goes to bash's stdin
as one line: `eval` of the quoted command with stdin from /dev/null, then a
printf of an end marker with a fresh random nonce, the return code and the cwd.
bash reads the whole line before it runs any of it, so while the command runs
the nonce is nowhere it can read (not on bash's stdin, not in a variable), and
it can't print a marker that ends its own output early. On timeout the process
group is killed and bash restarted in the last known cwd. Once closed, the
shell stays closed: run() refuses rather than start a new bash.

On native Windows the shell is Git Bash: a bare "bash" there can start WSL's launcher instead (Windows
searches System32 before PATH), there are no process groups (taskkill /T kills the tree), and the cwd is
read back as a Windows path so bash can be restarted in it.
"""

from __future__ import annotations

import contextlib
import os
import queue
import secrets
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

WINDOWS = os.name == "nt"
START_TIMEOUT_S = 30.0  # how long a new shell may take to answer through its sandbox before commands go to it anyway

ENV_ALLOW = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "TERM",
    "TMPDIR",
    "TZ",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    # what Windows programs need to start (Git Bash and anything it runs)
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
}


class PersistentShell:
    def __init__(
        self,
        cwd: str,
        env: dict | None = None,
        wrapper: list[str] | None = None,
        timeout: int = 120,
        max_output: int = 10000,
    ):
        self.cwd = self.home = str(Path(cwd).resolve())  # home: where bash restarts if the last cwd is gone
        self.closed = False  # set by close(): no command runs and no bash starts after it
        # an allowlisted environment, not the harness's: API keys and tokens stay out of the agent's shell
        self.env = {k: v for k, v in os.environ.items() if k in ENV_ALLOW or k.startswith("LC_")} | (env or {})
        self.wrapper = wrapper or []
        self.timeout = timeout
        self.hold = lambda: False  # true while the broker handles one of this shell's calls (set by the runtime)
        self.max_hold = 600.0  # the broker's call deadline (policy.CALL_DEADLINE_S, set by the runtime)
        # tells the broker the running command was killed; returns a note for the agent or "" (set by the runtime)
        self.on_kill = lambda: ""
        self.max_output = max_output
        self.proc: subprocess.Popen | None = None
        self.start_error, self._ready = "", ""  # why the sandbox couldn't start; the ready line _await_ready awaits
        self._q: queue.Queue = queue.Queue()
        self._start()

    def _start(self) -> None:
        if self.closed:
            return
        if not os.path.isdir(self.cwd):  # removed since bash reported it: restart where the shell began
            self.cwd = self.home
        try:
            self.proc = subprocess.Popen(
                [*self.wrapper, _bash(), "--noprofile", "--norc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=self.env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as e:  # no bash, or no directory to start in: run says so, and tries again next command
            self.proc, self.start_error = None, f"{type(e).__name__}: {e}"
            return
        self._q = queue.Queue()
        q, out = self._q, self.proc.stdout

        def pump():
            for line in out:
                q.put(line)
            q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        self.start_error = self._await_ready() if self.wrapper else ""

    def _await_ready(self) -> str:
        """Wait for bash to answer through the sandbox wrapper. Returns "" once it does, or what the wrapper printed
        when it exited first (bwrap's own error, say): then no command can run, and run says so plainly."""
        marker = self._ready = f"__OCB_READY_{uuid.uuid4().hex}__"  # run drops it if it comes late
        with contextlib.suppress(OSError):  # the wrapper may have exited already: the pump then ends the queue
            self.proc.stdin.write(f"printf '%s\\n' {marker}\n")
            self.proc.stdin.flush()
        lines: list[str] = []
        deadline = time.monotonic() + START_TIMEOUT_S
        while True:
            try:
                line = self._q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:  # slow, not failed: the first command waits for it as usual
                return ""
            if line is None:
                return "".join(lines).strip() or f"the sandbox exited with code {self.proc.wait()}"
            if line.rstrip("\n") == marker:
                return ""
            lines.append(line)

    def run(self, command: str, timeout: int | None = None) -> dict:
        """Run one action. Returns {output, returncode, timed_out}, plus closed=True once the shell is closed."""
        if self.closed:
            return dict(CLOSED)
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        if self.proc is None:  # tried again for each command: whatever stopped it may have been fixed since
            msg = f"[shell error] bash couldn't start, so the command did not run: {self.start_error}"
            return {"output": _truncate(msg, self.max_output), "returncode": -1, "timed_out": False}
        if self.start_error:  # tried again for each command: whatever stopped it may have been fixed since
            msg = f"[sandbox error] the shell's sandbox couldn't start, so the command did not run: {self.start_error}"
            return {"output": _truncate(msg, self.max_output), "returncode": -1, "timed_out": False}
        marker = f"__OCB_{secrets.token_hex(16)}__"  # a fresh nonce per command
        # the command goes in on stdin, quoted for eval in this shell (so cd and exports carry over): no script file
        # on a path the agent can write, which a background job of its could swap for a link. One line, so bash
        # has read the marker before the command starts: none of it is left on stdin for the command to read
        try:
            self.proc.stdin.write(
                f"eval {_q(command)} < /dev/null; __ocb_rc=$?; printf '\\n{marker} %s %s\\n' \"$__ocb_rc\" {_PWD}\n"
            )
            self.proc.stdin.flush()
        except OSError:  # bash exited since the check above (or close() killed it): the command didn't run
            self.proc = None
            return dict(CLOSED) if self.closed else {"output": "", "returncode": -1, "timed_out": False}
        lines: list[str] = []
        deadline = time.monotonic() + (timeout or self.timeout)  # a deadline from the start, not an idle timeout
        hard = deadline + self.max_hold  # holds can't stretch it forever (e.g. a background loop of calls)
        while True:
            try:
                line = self._q.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                if self.hold() and time.monotonic() < hard:  # waiting on the broker (an approval, the gate)
                    deadline = time.monotonic() + 1.0
                    continue
                note = self._kill()  # the broker names a call it was sending at the kill: its result is lost
                out = _truncate("".join(lines), self.max_output) + (f"\n{note}" if note else "")
                return {"output": out, "returncode": -1, "timed_out": True}
            if line is None:  # shell exited (e.g. the action ran `exit`), or close() killed it
                self.proc = None
                if self.closed:
                    return dict(CLOSED)
                return {"output": _truncate("".join(lines), self.max_output), "returncode": -1, "timed_out": False}
            if line.startswith(marker + " "):
                _, rc, cwd = line.rstrip("\n").split(" ", 2)
                cwd = os.path.normpath(cwd) if WINDOWS else cwd
                if os.path.isdir(cwd):  # a cwd deleted under the shell is kept out: a restart couldn't start in it
                    self.cwd = cwd
                out = "".join(lines)
                if out.endswith("\n"):
                    out = out[:-1]  # the newline printf put before the marker
                return {"output": _truncate(out, self.max_output), "returncode": int(rc), "timed_out": False}
            if line.rstrip("\n") != self._ready:
                lines.append(line)

    def interrupt(self) -> None:
        """Ctrl-C: kill the running action, keep a fresh shell in the same cwd."""
        self._kill()

    def _kill(self) -> str:
        note = self.on_kill() or ""
        if self.proc and self.proc.poll() is None:
            _kill_tree(self.proc.pid)
        self.proc = None
        self._start()
        return note

    def close(self) -> None:
        """Kill bash for good: a loop still running on this shell (a turn the caller abandoned) gets CLOSED from
        run() instead of a new bash."""
        self.closed = True
        if self.proc and self.proc.poll() is None:
            _kill_tree(self.proc.pid)


CLOSED = {
    "output": "[shell closed] the session's shell was closed, so the command did not run",
    "returncode": -1,
    "timed_out": False,
    "closed": True,
}

_PWD = '"$(pwd -W)"' if WINDOWS else '"$PWD"'  # Git Bash's $PWD is /c/...; pwd -W gives C:/...


def _bash() -> str:
    """The bash to start: on Windows, the first bash.exe on PATH outside the Windows directory (Git Bash)."""
    if not WINDOWS:
        return "bash"
    windir = os.path.normcase(os.environ.get("SYSTEMROOT", r"C:\Windows"))
    for d in os.environ.get("PATH", "").split(os.pathsep):
        exe = shutil.which("bash", path=d) if d else None
        if exe and not os.path.normcase(os.path.abspath(exe)).startswith(windir):
            return exe
    raise RuntimeError("no bash on PATH: install Git for Windows (Git Bash), or run in WSL2")


def _kill_tree(pid: int) -> None:
    """Kill the shell and everything it started: its process group, and every descendant, including those that
    left the group with setsid (on Linux the sandbox's PID namespace also takes them down with the shell)."""
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
        return
    procs = _descendants(pid)
    for p in [pid, *procs]:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(p, signal.SIGKILL)  # p's own group, if it leads one (a setsid child does)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, signal.SIGKILL)


def _descendants(pid: int) -> list[int]:
    """Every process below pid, from one `ps` snapshot (children by parent pid)."""
    try:
        out = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid="], check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no ps: the process group still goes
        return []
    children: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            found.append(c)
            todo.append(c)
    return found


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _truncate(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    half = n // 2
    return s[:half] + f"\n[... {len(s) - n} characters elided; narrow the command (head, grep, tail) ...]\n" + s[-half:]
