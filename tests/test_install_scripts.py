"""install.sh: it installs the release this checkout is (pyproject.toml), skips a rerun only when uv's tool is exactly
that requirement, brings a pinned and checksummed uv, and fails with exactly one stderr line.

It runs against stubs (uv, curl, uname) on macOS and Linux; native Windows isn't supported by the script (WSL2 is)."""

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VERSION = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
PINNED = f"olas-ocb-agent[facilitator]=={VERSION}"
LISTED = f"olas-ocb-agent v{VERSION} [required: =={VERSION}] [extras: facilitator]"  # how uv lists PINNED
UV_VERSION = re.search(r'^UV_VERSION="([^"]+)"', (ROOT / "install.sh").read_text(), re.MULTILINE).group(1)
EXCLUDE_NEWER = re.search(r'^EXCLUDE_NEWER="([^"]+)"', (ROOT / "install.sh").read_text(), re.MULTILINE).group(1)
# How install.sh calls uv: no --force (a user's own ocb-agent or pearl is never replaced), deps frozen at the release.
INSTALL_FLAGS = (
    "tool install --python 3.12 --reinstall-package olas-ocb-agent "
    f"--exclude-newer {EXCLUDE_NEWER} --exclude-newer-package olas-ocb-agent=false"
)


def test_the_package_installs_exactly_two_commands():
    pyproject = (ROOT / "pyproject.toml").read_text()
    scripts = pyproject.split("[project.scripts]\n", 1)[1].split("\n\n", 1)[0]
    assert [ln.split(" =")[0] for ln in scripts.splitlines()] == ["ocb-agent", "pearl"]


def test_install_sh_defaults_to_this_version_and_pins_its_downloads():
    sh = (ROOT / "install.sh").read_text()
    assert re.findall(r'^DEFAULT_VERSION="([^"]+)"', sh, re.MULTILINE) == [VERSION], "install.sh: bump to pyproject"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", EXCLUDE_NEWER)
    # the versioned installer, checked before it runs; never astral.sh's latest
    assert re.search(r'^UV_INSTALLER_SHA256="[0-9a-f]{64}"', sh, re.MULTILINE) and "astral.sh/uv/install." not in sh


# ---- install.sh ------------------------------------------------------------------------------------
# A uv whose behaviour is set by files beside it: `mode` (ok | fail | nobin: installs no ocb-agent) and
# `version`. `tool install` logs its arguments, records the tool as uv lists it, and writes its
# venv's pyvenv.cfg on a managed Python 3.12 (under `python dir`).
_STUB_UV = r"""#!/bin/sh
here="$(dirname "$0")"
mode="$(cat "$here/mode" 2>/dev/null || echo ok)"
case "$1 ${2:-} ${3:-}" in
  "--version "*) echo "uv $(cat "$here/version" 2>/dev/null || echo 0.13.8) (stub)" ;;
  "tool dir --bin") echo "$STUB_BIN" ;;
  "tool dir ") echo "$STUB_TOOLS" ;;
  "python dir ") echo "$STUB_PYTHONS" ;;
  "tool list "*) cat "$STUB_LIST" 2>/dev/null; exit 0 ;;
  "tool install "*)
    echo "$0 $*" >> "$STUB_LOG"
    echo "Resolved 15 packages" >&2
    if [ "$mode" = fail ]; then echo "error: stub failure" >&2; echo "  cause: no such version" >&2; exit 2; fi
    [ "$mode" != nobin ] || exit 0
    case " $* " in *" --force "*) ;; *)  # like uv: never replace an executable that isn't this tool's
      for e in ocb-agent pearl; do
        if [ -e "$STUB_BIN/$e" ] && ! grep -q stub-ocb "$STUB_BIN/$e"; then
          echo "error: Executable already exists: $e (use \`--force\` to overwrite)" >&2; exit 2
        fi
      done ;;
    esac
    for spec in "$@"; do :; done
    case "$spec" in
      "olas-ocb-agent[facilitator]=="*) v="${spec#*==}"; req="==$v" ;;
      *) v="$NEW_VERSION"; req="$spec" ;;
    esac
    printf 'olas-ocb-agent v%s [required: %s] [extras: facilitator]\n- ocb-agent\n' "$v" "$req" > "$STUB_LIST"
    printf '#!/bin/sh\n# stub-ocb\necho "ocb-agent %s"\n' "$v" > "$STUB_BIN/ocb-agent"
    chmod +x "$STUB_BIN/ocb-agent"
    mkdir -p "$STUB_TOOLS/olas-ocb-agent"
    printf 'home = %s/cpython-3.12-linux-x86_64-gnu/bin\nversion_info = 3.12.16\n' "$STUB_PYTHONS" \
      > "$STUB_TOOLS/olas-ocb-agent/pyvenv.cfg" ;;
  *) exit 64 ;;
esac
"""
# Astral's installer, faked: puts a working uv in UV_UNMANAGED_INSTALL. It refuses to run in a mode that would write
# an install receipt (repointing a user's own uv) or edit PATH: what 0.13.8's installer does without these.
# Like the real one, it checks the uv archive with `sha256sum -b` and would skip the check without it: here it fails.
_FAKE_INSTALLER = """#!/bin/sh
[ -n "$UV_UNMANAGED_INSTALL" ] && [ "$UV_DISABLE_UPDATE" = 1 ] || { echo "would write a receipt" >&2; exit 3; }
command -v sha256sum >/dev/null || { echo "skipping sha256 checksum verification" >&2; exit 4; }
sha256sum -b "$0" | grep -q '^[0-9a-f]\\{64\\}' || { echo "sha256sum doesn't work" >&2; exit 5; }
d="$UV_UNMANAGED_INSTALL"
mkdir -p "$d" && rm -f "$d/version" "$d/mode"  # a new uv: the pinned one
cp "$STUB_UV_SRC" "$d/uv" && chmod +x "$d/uv"
echo "everything's installed!"
"""
_STUB_CURL = """#!/bin/sh
echo "$*" >> "$CURL_LOG"
while [ $# -gt 0 ]; do [ "$1" != -o ] || out="$2"; shift; done
cp "$FAKE_INSTALLER" "$out"
"""


def _exe(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _mirror(dirs, dest: Path, exclude) -> None:
    """Link every entry of `dirs` into `dest` but `exclude`, the first directory winning as on PATH. lexists, not
    exists: a broken link (as CI's /bin has) doesn't exist, and linking its name a second time would raise."""
    dest.mkdir()
    for d in dirs:
        for entry in d.iterdir():
            link = dest / entry.name
            if entry.name not in exclude and not os.path.lexists(link):
                link.symlink_to(entry)


class Sh:
    """install.sh in a scratch HOME: its own uv is a current stub, PATH holds a decoy uv (never to be run) and a
    stub curl."""

    def __init__(self, tmp: Path):
        self.tmp, self.bin, self.own = tmp, tmp / "bin", tmp / "data" / "olas-ocb-agent" / "bin"
        self.path_dir, self.tmpdir = tmp / "path", tmp / "t"
        self.bin.mkdir()
        self.tmpdir.mkdir()
        self.stub_src = _exe(tmp / "stub-uv", _STUB_UV)
        self.own_uv = _exe(self.own / "uv", _STUB_UV)
        (self.own / "version").write_text(UV_VERSION)
        self.decoy_log = tmp / "decoy.log"
        _exe(self.path_dir / "uv", f'#!/bin/sh\necho "$*" >> "{self.decoy_log}"\necho "$STUB_BIN"\n')
        _exe(self.path_dir / "curl", _STUB_CURL)
        self.installer = tmp / "fake-installer.sh"
        self.installer.write_text(_FAKE_INSTALLER)
        self.script = ROOT / "install.sh"
        self.log, self.list, self.curl_log = tmp / "installs.log", tmp / "tools.txt", tmp / "curl.log"
        self.path = f"{self.path_dir}{os.pathsep}/usr/bin{os.pathsep}/bin"

    def without_sha256sum(self, shasum: bool) -> None:
        """PATH as the system's, minus sha256sum (as on macOS), and minus shasum too unless `shasum`."""
        real = shutil.which("sha256sum")
        if shasum and real is None:
            pytest.skip("needs a real sha256sum behind the stub shasum")
        sys_dir = self.tmp / "sys"
        _mirror((Path("/usr/bin"), Path("/bin")), sys_dir, exclude=("sha256sum", "shasum"))
        if shasum:  # `shasum -a 256 [-b] FILE`, as macOS has it
            _exe(sys_dir / "shasum", f'#!/bin/sh\n[ "$1" = -a ] && shift 2\nexec "{real}" "$@"\n')
        self.path = f"{self.path_dir}{os.pathsep}{sys_dir}"

    def trust_fake_installer(self) -> None:
        """A copy of install.sh pinned to the fake installer's sha256 (the real one is Astral's)."""
        digest = hashlib.sha256(self.installer.read_bytes()).hexdigest()
        text = re.sub(
            r'^UV_INSTALLER_SHA256="[0-9a-f]{64}"',
            f'UV_INSTALLER_SHA256="{digest}"',
            self.script.read_text(),
            flags=re.MULTILINE,
        )
        self.script = self.tmp / "install.sh"
        self.script.write_text(text)

    def installs(self) -> list[str]:
        out = self.log.read_text().splitlines() if self.log.exists() else []
        self.log.unlink(missing_ok=True)
        return out

    def run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        base = {
            "PATH": self.path,
            "HOME": str(self.tmp),
            "TMPDIR": str(self.tmpdir),
            "XDG_DATA_HOME": str(self.tmp / "data"),
            "STUB_BIN": str(self.bin),
            "STUB_LOG": str(self.log),
            "STUB_LIST": str(self.list),
            "STUB_TOOLS": str(self.tmp / "tools"),
            "STUB_PYTHONS": str(self.tmp / "pythons"),
            "STUB_UV_SRC": str(self.stub_src),
            "FAKE_INSTALLER": str(self.installer),
            "CURL_LOG": str(self.curl_log),
            "NEW_VERSION": VERSION,
        }
        return subprocess.run(
            ["sh", str(self.script), *args],
            env={**base, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def error(self, r: subprocess.CompletedProcess) -> str:
        """The failure contract: exit non-zero, nothing on stdout, exactly one stderr line."""
        assert r.returncode != 0 and r.stdout == ""
        assert len(r.stderr.splitlines()) == 1, r.stderr
        assert r.stderr.startswith("ocb-install: error: ")
        return r.stderr


@pytest.fixture
def sh(tmp_path):
    if sys.platform == "win32" or not shutil.which("sh"):
        pytest.skip("install.sh is for macOS, Linux and WSL2")
    return Sh(tmp_path)


def test_install_then_a_rerun_changes_nothing(sh):
    r = sh.run()
    assert (r.returncode, r.stdout, r.stderr) == (0, f"{sh.bin / 'ocb-agent'}\n", "")  # stderr isn't a terminal
    [call] = sh.installs()
    assert call.endswith(f"{INSTALL_FLAGS} {PINNED}") and " --force " not in call
    assert call.startswith(str(sh.own_uv)) and list(sh.tmpdir.iterdir()) == []  # the log is removed on success
    assert not sh.decoy_log.exists()  # a uv on PATH is never run
    r = sh.run()
    assert (r.returncode, r.stdout, sh.installs()) == (0, f"{sh.bin / 'ocb-agent'}\n", [])


@pytest.mark.parametrize("name", ["pearl", "ocb-agent"])
def test_a_users_own_executable_is_never_replaced(sh, name):
    mine = "#!/bin/sh\necho mine\n"
    _exe(sh.bin / name, mine)
    err = sh.error(sh.run())  # one line: the cause (another tool's executable) and the fix
    assert f"{sh.bin / name} already exists and isn't olas-ocb-agent's" in err
    assert f"the unrelated '{name}' package from PyPI" in err and "uv tool uninstall" in err and "rerun" in err
    assert (sh.bin / name).read_text() == mine and len(sh.installs()) == 1


def test_without_sha256sum_the_uv_installer_gets_one_from_shasum(sh):
    sh.own_uv.unlink()
    sh.trust_fake_installer()
    sh.without_sha256sum(shasum=True)
    r = sh.run()  # the fake installer exits 4 if it would skip its archive check
    assert r.returncode == 0, r.stderr
    assert sh.installs()[0].startswith(str(sh.own_uv)) and list(sh.tmpdir.iterdir()) == []  # the shim is removed


def test_without_sha256sum_or_shasum_nothing_is_run(sh):
    sh.own_uv.unlink()
    sh.trust_fake_installer()
    sh.without_sha256sum(shasum=False)
    assert "neither sha256sum nor shasum is available" in sh.error(sh.run())
    assert not sh.own_uv.exists() and sh.installs() == []


@pytest.mark.parametrize(
    "listed",
    [
        f"olas-ocb-agent v{VERSION} [required: git+https://github.com/valory-xyz/olas-ocb-agent@a-branch] "
        "[extras: facilitator]",  # a branch build reporting the same version
        f"olas-ocb-agent v{VERSION} [required: =={VERSION}]",  # without the facilitator extra
        "olas-ocb-agent v0.0.1 [required: ==0.0.1] [extras: facilitator]",  # another release
    ],
)
def test_anything_but_exactly_the_pinned_requirement_is_reinstalled(sh, listed):
    sh.run()
    sh.installs()
    sh.list.write_text(listed + "\n- ocb-agent\n")
    r = sh.run()
    assert r.returncode == 0 and [c.split()[-1] for c in sh.installs()] == [PINNED]
    assert sh.list.read_text().startswith(LISTED)


@pytest.mark.parametrize(
    "cfg",
    [
        "home = /usr/bin\nversion_info = 3.12.3\n",  # a system Python 3.12, from someone else's uv
        "home = {pythons}/cpython-3.10-linux-x86_64-gnu/bin\nversion_info = 3.10.16\n",  # managed, not 3.12
        None,  # no venv config at all
    ],
)
def test_a_matching_tool_on_another_interpreter_is_reinstalled(sh, cfg):
    sh.run()
    sh.installs()
    venv_cfg = sh.tmp / "tools" / "olas-ocb-agent" / "pyvenv.cfg"
    if cfg is None:
        venv_cfg.unlink()
    else:
        venv_cfg.write_text(cfg.format(pythons=sh.tmp / "pythons"))
    assert sh.run().returncode == 0 and [c.split()[-1] for c in sh.installs()] == [PINNED]
    assert "version_info = 3.12.16" in venv_cfg.read_text()


def test_a_missing_ocb_agent_is_reinstalled_even_if_listed(sh):
    sh.list.write_text(LISTED + "\n")
    assert sh.run().returncode == 0 and len(sh.installs()) == 1


def test_a_listed_ocb_agent_that_fails_to_start_is_reinstalled(sh):
    sh.run()
    sh.installs()
    _exe(sh.bin / "ocb-agent", "#!/bin/sh\n# stub-ocb\nexit 126\n")  # e.g. its venv's Python is gone
    r = sh.run()
    assert (r.returncode, r.stdout, len(sh.installs())) == (0, f"{sh.bin / 'ocb-agent'}\n", 1)
    assert subprocess.run([sh.bin / "ocb-agent", "--version"], capture_output=True, check=False).returncode == 0


def test_a_custom_spec_always_reinstalls(sh):
    spec = "olas-ocb-agent[facilitator] @ https://github.com/valory-xyz/olas-ocb-agent/archive/abc.tar.gz"
    for _ in range(2):
        r = sh.run(OCB_AGENT_SPEC=spec)
        assert r.returncode == 0 and [c.endswith(spec) for c in sh.installs()] == [True]


@pytest.mark.parametrize(
    ("args", "env", "want"),
    [
        (("0.9.0",), {}, "==0.9.0"),
        (("v0.9.0",), {}, "==0.9.0"),  # a tag-style version is normalised
        ((), {"OCB_AGENT_VERSION": "v1.2.3"}, "==1.2.3"),
        (("2.0.0",), {"OCB_AGENT_VERSION": "1.2.3"}, "==2.0.0"),  # the argument wins
    ],
)
def test_version_argument_and_environment(sh, args, env, want):
    assert sh.run(*args, **env).returncode == 0
    assert [c.split()[-1] for c in sh.installs()] == [f"olas-ocb-agent[facilitator]{want}"]


def test_a_version_that_isnt_one_is_refused(sh):
    assert "not a version" in sh.error(sh.run("1.0;rm"))
    assert sh.installs() == []


def test_a_failed_install_is_one_stderr_line_with_uvs_details_in_the_log(sh):
    (sh.own / "mode").write_text("fail")
    err = sh.error(sh.run())
    assert "couldn't install " + PINNED in err
    log = Path(err.rsplit("(details: ", 1)[1].rstrip(")\n"))
    assert "error: stub failure" in log.read_text() and "cause: no such version" in log.read_text()


def test_no_ocb_agent_after_installing_is_an_error(sh):
    (sh.own / "mode").write_text("nobin")
    assert f"ocb-agent isn't at {sh.bin / 'ocb-agent'} after installing" in sh.error(sh.run())


def test_unsupported_os(sh):
    _exe(sh.path_dir / "uname", "#!/bin/sh\necho MINGW64_NT-10.0\n")
    assert "unsupported OS MINGW64_NT-10.0; on Windows, run it in WSL2" in sh.error(sh.run())


def test_without_its_uv_it_installs_the_pinned_one_after_checking_its_installer(sh):
    sh.own_uv.unlink()
    sh.trust_fake_installer()
    r = sh.run()
    assert r.returncode == 0, r.stderr
    assert f"https://astral.sh/uv/{UV_VERSION}/install.sh" in sh.curl_log.read_text()
    assert sh.installs()[0].startswith(str(sh.own_uv)) and not sh.decoy_log.exists()
    sh.curl_log.unlink()
    assert sh.run().returncode == 0 and not sh.curl_log.exists()  # the own copy is reused


def test_an_installer_with_another_checksum_is_not_run(sh):
    sh.own_uv.unlink()
    err = sh.error(sh.run())  # the real pin, the fake installer
    assert f"the uv installer from https://astral.sh/uv/{UV_VERSION}/install.sh has sha256" in err
    assert not sh.own_uv.exists() and sh.installs() == [] and not sh.decoy_log.exists()


def test_an_own_uv_of_another_version_is_replaced(sh):
    sh.trust_fake_installer()
    (sh.own / "version").write_text("0.1.0")
    assert sh.run().returncode == 0 and sh.curl_log.exists()
    assert sh.installs()[0].startswith(str(sh.own_uv)) and not sh.decoy_log.exists()


def test_mirroring_bin_dirs_tolerates_broken_and_repeated_links(tmp_path):
    """CI runners have broken links in /bin (e.g. /bin/grub-ntldr-img), and /bin may be /usr/bin again."""
    if sys.platform == "win32":
        pytest.skip("symlinks: POSIX only")
    a, b, dest = tmp_path / "usr-bin", tmp_path / "bin", tmp_path / "sys"
    for d in (a, b):
        d.mkdir()
        (d / "broken").symlink_to(tmp_path / "nowhere")
        _exe(d / "tool", "#!/bin/sh\n")
    (a / "sha256sum").write_text("")
    _mirror((a, b), dest, exclude=("sha256sum",))
    assert sorted(p.name for p in dest.iterdir()) == ["broken", "tool"]
    assert os.readlink(dest / "tool") == str(a / "tool")  # the first directory wins, as on PATH
