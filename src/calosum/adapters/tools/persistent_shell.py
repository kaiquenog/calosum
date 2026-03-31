from __future__ import annotations

import asyncio
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
import tempfile

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
    Ferramenta de shell persistente que mantém o processo vivo entre chamadas da mesma sessão.
    """
    def __init__(self, ttl_seconds: float = 600.0) -> None:
        self.sessions: dict[str, ShellSession] = {}
        self.ttl_seconds = ttl_seconds
        self.schema = ToolSchema(
            name="execute_bash_persistent",
            description="Execute bash command in a persistent session. Maintains state (cd, env vars) between calls.",
            parameters={"command": "string"},
            required_permissions=["shell"],
            needs_approval=True,
        )

    async def execute(self, payload: dict[str, object], session_id: str | None = None) -> str:
        if not session_id:
            return "Error: session_id is required for persistent shell."
        
        command = str(payload.get("command", "")).strip()
        if not command:
            return "No command provided."

        session = await self._get_or_create_session(session_id)
        return await self._run_command(session, command)

    async def _get_or_create_session(self, session_id: str) -> ShellSession:
        self._cleanup_expired()
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if session.process.returncode is None:
                session.last_activity = asyncio.get_event_loop().time()
                return session
            else:
                logger.warning(f"Session {session_id} process died. Restarting.")

        sandbox_dir = Path(tempfile.gettempdir()) / "calosum_sandbox" / session_id
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(sandbox_dir),
            env={"PS1": "", "TERM": "dumb"}
        )

        session = ShellSession(
            process=process,
            session_id=session_id,
            sandbox_dir=sandbox_dir,
            last_activity=asyncio.get_event_loop().time()
        )
        self.sessions[session_id] = session
        return session

    async def _run_command(self, session: ShellSession, command: str) -> str:
        delimiter = str(uuid.uuid4())
        
        # We wrap the command to output a delimiter on its own line.
        full_command = f"{command}\necho ''\necho {delimiter}\n"
        
        try:
            assert session.process.stdin is not None
            session.process.stdin.write(full_command.encode())
            await session.process.stdin.drain()

            output_lines = []
            assert session.process.stdout is not None
            
            # Read until delimiter
            while True:
                line_bytes = await session.process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8', errors='replace').rstrip('\n\r')
                if line == delimiter:
                    break
                output_lines.append(line)

            # Cleanup the empty line from echo '' if first one is empty after cat
            if output_lines and not output_lines[-1].strip() and len(output_lines) > 1:
                output_lines.pop()

            # Check if there's any immediate error in stderr (not perfect as it depends on stderr buffering)
            # but let's at least try a quick read if possible.
            # In a real shell, we'd probably redirect 2>&1 in the full_command.
            
            return "\n".join(output_lines) if output_lines else "Command executed."
            
        except Exception as e:
            logger.error(f"Error in persistent shell: {e}")
            return f"Bash execution failed: {e}"

    def _cleanup_expired(self) -> None:
        now = asyncio.get_event_loop().time()
        to_delete = []
        for sid, session in self.sessions.items():
            if now - session.last_activity > self.ttl_seconds:
                to_delete.append(sid)
        
        for sid in to_delete:
            session = self.sessions.pop(sid)
            try:
                session.process.terminate()
            except Exception:
                pass
