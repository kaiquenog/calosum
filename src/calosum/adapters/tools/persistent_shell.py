from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from calosum.shared.tools import ToolSchema


logger = logging.getLogger(__name__)


@dataclass
class ShellSession:
    process: asyncio.subprocess.Process
    session_id: str
    sandbox_dir: Path
    last_activity: float


class PersistentShellTool:
    """
    Shell persistente por sessão, com timeout, exit code e ciclo de vida explícito.
    """

    def __init__(self, ttl_seconds: float = 600.0, command_timeout_seconds: float = 20.0) -> None:
        self.sessions: dict[str, ShellSession] = {}
        self.ttl_seconds = ttl_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.schema = ToolSchema(
            name="execute_bash_persistent",
            description="Execute bash command in a persistent session. Maintains state (cd, env vars) between calls.",
            parameters={"command": "string"},
            required_permissions=["shell"],
            needs_approval=True,
        )

    async def execute(self, payload: dict[str, Any], session_id: str | None = None, **_: Any) -> str:
        if not session_id:
            return "Error: session_id is required for persistent shell."
        command = str(payload.get("command", "")).strip()
        if not command:
            return "No command provided."
        session = await self._get_or_create_session(session_id)
        response = await self._run_command(session, command)
        return json.dumps(response, ensure_ascii=False)

    async def close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if not session:
            return
        try:
            session.process.terminate()
            await asyncio.wait_for(session.process.wait(), timeout=1.0)
        except Exception:
            try:
                session.process.kill()
            except Exception:
                pass

    async def _get_or_create_session(self, session_id: str) -> ShellSession:
        await self._cleanup_expired()
        existing = self.sessions.get(session_id)
        if existing and existing.process.returncode is None:
            existing.last_activity = time.monotonic()
            return existing
        sandbox_dir = Path(tempfile.gettempdir()) / "calosum_sandbox" / session_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(sandbox_dir),
            env={"PS1": "", "TERM": "dumb"},
        )
        session = ShellSession(
            process=process,
            session_id=session_id,
            sandbox_dir=sandbox_dir,
            last_activity=time.monotonic(),
        )
        self.sessions[session_id] = session
        return session

    async def _run_command(self, session: ShellSession, command: str) -> dict[str, Any]:
        marker = f"__CALOSUM_{uuid.uuid4().hex}__"
        out_begin = f"{marker}_OUT_BEGIN"
        out_end = f"{marker}_OUT_END"
        err_begin = f"{marker}_ERR_BEGIN"
        err_end = f"{marker}_ERR_END"
        exit_marker = f"{marker}_EXIT:"
        started = time.perf_counter()
        script = "\n".join(
            [
                "__calosum_out=$(mktemp)",
                "__calosum_err=$(mktemp)",
                "{",
                command,
                "} >\"$__calosum_out\" 2>\"$__calosum_err\"",
                "__calosum_exit=$?",
                f"echo \"{out_begin}\"",
                "cat \"$__calosum_out\"",
                f"echo \"{out_end}\"",
                f"echo \"{err_begin}\"",
                "cat \"$__calosum_err\"",
                f"echo \"{err_end}\"",
                f"echo \"{exit_marker}$__calosum_exit\"",
                "rm -f \"$__calosum_out\" \"$__calosum_err\"",
            ]
        ) + "\n"
        try:
            if session.process.stdin is None or session.process.stdout is None:
                raise RuntimeError("persistent shell process pipes are unavailable")
            session.process.stdin.write(script.encode("utf-8"))
            await session.process.stdin.drain()
            parser = self._collect_command_output(
                session.process.stdout,
                out_begin=out_begin,
                out_end=out_end,
                err_begin=err_begin,
                err_end=err_end,
                exit_marker=exit_marker,
            )
            stdout_text, stderr_text, exit_code = await asyncio.wait_for(
                parser,
                timeout=self.command_timeout_seconds,
            )
        except TimeoutError:
            return {
                "status": "timeout",
                "exit_code": None,
                "stdout": "",
                "stderr": f"Command timed out after {self.command_timeout_seconds} seconds.",
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "session_id": session.session_id,
                "cwd": str(session.sandbox_dir),
            }
        except Exception as exc:
            logger.error("Error in persistent shell: %s", exc)
            return {
                "status": "error",
                "exit_code": None,
                "stdout": "",
                "stderr": f"Bash execution failed: {exc}",
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "session_id": session.session_id,
                "cwd": str(session.sandbox_dir),
            }

        session.last_activity = time.monotonic()
        return {
            "status": "ok" if exit_code == 0 else "error",
            "exit_code": exit_code,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "session_id": session.session_id,
            "cwd": str(session.sandbox_dir),
        }

    async def _collect_command_output(
        self,
        stdout: asyncio.StreamReader,
        *,
        out_begin: str,
        out_end: str,
        err_begin: str,
        err_end: str,
        exit_marker: str,
    ) -> tuple[str, str, int]:
        out_lines: list[str] = []
        err_lines: list[str] = []
        mode: str | None = None
        exit_code = 1
        while True:
            raw = await stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
            if line == out_begin:
                mode = "out"
                continue
            if line == out_end:
                mode = None
                continue
            if line == err_begin:
                mode = "err"
                continue
            if line == err_end:
                mode = None
                continue
            if line.startswith(exit_marker):
                tail = line[len(exit_marker) :].strip()
                try:
                    exit_code = int(tail)
                except ValueError:
                    exit_code = 1
                break
            if mode == "out":
                out_lines.append(line)
            elif mode == "err":
                err_lines.append(line)
        return "\n".join(out_lines).strip(), "\n".join(err_lines).strip(), exit_code

    async def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [sid for sid, session in self.sessions.items() if now - session.last_activity > self.ttl_seconds]
        for sid in expired:
            await self.close_session(sid)
