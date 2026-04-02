from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(slots=True)
class JEPARSConfig:
    """Configuration for JEPA-RS manager."""
    cache_dir: Path = Path.home() / ".calosum" / "jepa-rs"
    version_manifest_url: str = (
        "https://raw.githubusercontent.com/calosum/jepa-rs-releases/main/manifest.json"
    )
    timeout: int = 30


class JEPARSManager:
    """
    Manages download and versioning of jepa-rs binary by architecture and OS
    using SHA hash for reproducible executions.
    """

    def __init__(self, config: JEPARSConfig | None = None) -> None:
        self.config = config or JEPARSConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_platform_key(self) -> str:
        """Generate platform key for jepa-rs binary selection."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        # Normalize architecture names
        if machine in ("x86_64", "amd64"):
            arch = "x86_64"
        elif machine in ("arm64", "aarch64"):
            arch = "aarch64"
        else:
            arch = machine

        # Handle different OS names
        if system == "darwin":
            system = "apple-darwin"
        elif system == "linux":
            system = "unknown-linux-gnu"
        elif system == "windows":
            system = "pc-windows-msvc"
        else:
            system = f"{system}-unknown"

        return f"{arch}-{system}"

    def _download_manifest(self) -> dict[str, Any] | None:
        """Download version manifest from GitHub releases."""
        try:
            response = requests.get(
                self.config.version_manifest_url,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def _get_latest_version_info(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """Extract latest version info for current platform."""
        platform_key = self._get_platform_key()
        versions = manifest.get("versions", [])

        # Find latest version for this platform
        platform_versions = [
            v for v in versions if v.get("platform") == platform_key
        ]

        if not platform_versions:
            return None

        # Sort by version (semver) and take latest
        platform_versions.sort(
            key=lambda v: tuple(map(int, v["version"].split("."))),
            reverse=True,
        )
        return platform_versions[0]

    def _verify_sha256(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file SHA256 hash."""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest() == expected_hash
        except Exception:
            return False

    def ensure_jepa_rs(self, force_update: bool = False) -> Path:
        """
        Ensure jepa-rs binary is available and up to date.
        
        Args:
            force_update: Force re-download even if cached version exists
            
        Returns:
            Path to the jepa-rs binary
            
        Raises:
            RuntimeError: If binary cannot be obtained or verified
        """
        binary_name = "jepa-rs"
        if platform.system().lower() == "windows":
            binary_name += ".exe"

        binary_path = self.config.cache_dir / binary_name

        # Return existing binary if valid and not forcing update
        if binary_path.exists() and not force_update:
            # TODO: Could add version check here if we stored metadata
            return binary_path

        # Download manifest
        manifest = self._download_manifest()
        if manifest is None:
            raise RuntimeError("Failed to download jepa-rs version manifest")

        # Get latest version for platform
        version_info = self._get_latest_version_info(manifest)
        if version_info is None:
            raise RuntimeError(
                f"No jepa-rs version found for platform {self._get_platform_key()}"
            )

        version = version_info["version"]
        download_url = version_info["url"]
        expected_hash = version_info["sha256"]

        # Check if we have correct version cached
        if binary_path.exists() and not force_update:
            if self._verify_sha256(binary_path, expected_hash):
                return binary_path
            # Hash mismatch, re-download

        # Download binary
        try:
            response = requests.get(
                download_url,
                timeout=self.config.timeout,
                stream=True,
            )
            response.raise_for_status()

            # Write to temporary file first
            temp_path = binary_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify hash
            if not self._verify_sha256(temp_path, expected_hash):
                temp_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Downloaded jepa-rs binary hash mismatch. "
                    f"Expected: {expected_hash}"
                )

            # Move to final location and make executable
            temp_path.rename(binary_path)
            binary_path.chmod(0o755)  # rwxr-xr-x

            return binary_path

        except Exception as e:
            # Clean up on failure
            binary_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download jepa-rs: {e}") from e

    def get_version(self) -> str | None:
        """Get currently cached jepa-rs version."""
        manifest = self._download_manifest()
        if manifest is None:
            return None

        version_info = self._get_latest_version_info(manifest)
        if version_info is None:
            return None

        return version_info.get("version")