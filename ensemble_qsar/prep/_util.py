"""Small shared helpers for the prep steps: subprocess running + tool versions."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CmdResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def run_cmd(cmd: list[str], *, cwd: Path, log_path: Path | None = None) -> CmdResult:
    """Run `cmd` in `cwd`, capture output, optionally tee to `log_path`.

    Does not raise on non-zero exit; callers inspect `returncode` so the batch
    driver can record a failure and move on instead of aborting.
    """
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, check=False
    )
    dt = time.perf_counter() - t0
    if log_path is not None:
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
        )
    return CmdResult(cmd, proc.returncode, proc.stdout, proc.stderr, round(dt, 3))


def tool_version(exe: str) -> str:
    """Best-effort version string for an external tool (for the manifest)."""
    path = shutil.which(exe)
    if not path:
        return f"{exe}: MISSING"
    return f"{exe}: {path}"
