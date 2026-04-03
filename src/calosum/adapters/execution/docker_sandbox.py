from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class DockerToolSandbox:
    """
    Sandbox de execução baseado em Docker para isolamento real.
    Evita que o agente execute comandos diretamente no host do orquestrador.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        network_disabled: bool = True,
        mem_limit: str = "128m",
        cpu_quota: int = 50000, # 50% de um core
    ) -> None:
        self.image = image
        self.network_disabled = network_disabled
        self.mem_limit = mem_limit
        self.cpu_quota = cpu_quota
        self._container_name_by_session: dict[str, str] = {}

    async def execute_command(
        self,
        command: str,
        session_id: str = "global",
        timeout: float = 30.0,
        cwd: str = "/app",
    ) -> dict[str, Any]:
        container_name = await self._get_or_create_container(session_id)
        
        # Escapa o comando para ser executado via bash no container
        bash_command = ["docker", "exec", "-w", cwd, container_name, "bash", "-c", command]
        
        start_time = asyncio.get_event_loop().time()
        try:
            process = await asyncio.create_subprocess_exec(
                *bash_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            duration = asyncio.get_event_loop().time() - start_time
            
            return {
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "exit_code": process.returncode,
                "duration_ms": round(duration * 1000, 2),
                "status": "success" if process.returncode == 0 else "error",
            }
        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "exit_code": -1,
                "duration_ms": round(timeout * 1000, 2),
                "status": "timeout",
            }
        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return {
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "duration_ms": 0,
                "status": "failure",
            }

    async def _get_or_create_container(self, session_id: str) -> str:
        if session_id in self._container_name_by_session:
            name = self._container_name_by_session[session_id]
            # Verifica se ainda está rodando
            check = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await check.communicate()
            if stdout.decode().strip() == "true":
                return name
        
        # Cria novo container
        name = f"calosum-sandbox-{session_id}-{uuid.uuid4().hex[:8]}"
        net_flag = "--network=none" if self.network_disabled else ""
        
        create_cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--memory", self.mem_limit,
            "--cpu-quota", str(self.cpu_quota),
            "--rm",
            net_flag,
            self.image,
            "sleep", "infinity"
        ]
        
        # Remove empty strings from list
        create_cmd = [c for c in create_cmd if c]
        
        process = await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        
        self._container_name_by_session[session_id] = name
        return name

    async def stop_all(self) -> None:
        for name in self._container_name_by_session.values():
            process = await asyncio.create_subprocess_exec(
                "docker", "stop", "-t", "1", name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()
        self._container_name_by_session.clear()
