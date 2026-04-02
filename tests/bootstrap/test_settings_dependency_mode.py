from __future__ import annotations

import unittest
from unittest.mock import patch

from calosum.bootstrap.infrastructure.settings import (
    InfrastructureSettings,
    RuntimeDependencyMode,
)


class DependencyModeConsistencyTests(unittest.TestCase):
    def test_local_mode_requires_local_stack(self) -> None:
        with patch(
            "calosum.bootstrap.infrastructure.settings._missing_local_dependency_stack",
            return_value=["torch", "peft", "transformers"],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                InfrastructureSettings.from_sources(
                    environ={
                        "CALOSUM_DEPENDENCY_MODE": "local",
                        "CALOSUM_RIGHT_BACKEND": "",
                        "CALOSUM_VECTOR_QUANTIZATION": "none",
                    }
                )

        self.assertIn("CALOSUM_DEPENDENCY_MODE=local", str(ctx.exception))
        self.assertIn("pip install calosum[local]", str(ctx.exception))

    def test_api_mode_rejects_local_backend(self) -> None:
        with patch(
            "calosum.bootstrap.infrastructure.settings._missing_local_dependency_stack",
            return_value=[],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                InfrastructureSettings.from_sources(
                    environ={
                        "CALOSUM_DEPENDENCY_MODE": "api",
                        "CALOSUM_RIGHT_BACKEND": "huggingface",
                        "CALOSUM_VECTOR_QUANTIZATION": "none",
                    }
                )

        self.assertIn("CALOSUM_DEPENDENCY_MODE=api", str(ctx.exception))
        self.assertIn("right_hemisphere_backend=huggingface", str(ctx.exception))

    def test_api_mode_rejects_turboquant(self) -> None:
        with patch(
            "calosum.bootstrap.infrastructure.settings._missing_local_dependency_stack",
            return_value=[],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                InfrastructureSettings.from_sources(
                    environ={
                        "CALOSUM_DEPENDENCY_MODE": "api",
                        "CALOSUM_RIGHT_BACKEND": "",
                        "CALOSUM_VECTOR_QUANTIZATION": "turboquant",
                    }
                )

        self.assertIn("vector_quantization=turboquant", str(ctx.exception))

    def test_auto_mode_raises_when_local_feature_without_local_stack(self) -> None:
        with patch(
            "calosum.bootstrap.infrastructure.settings._missing_local_dependency_stack",
            return_value=["torch"],
        ):
            with self.assertRaises(RuntimeError) as ctx:
                InfrastructureSettings.from_sources(
                    environ={
                        "CALOSUM_DEPENDENCY_MODE": "auto",
                        "CALOSUM_RIGHT_BACKEND": "vjepa21",
                        "CALOSUM_VECTOR_QUANTIZATION": "none",
                    }
                )

        self.assertIn("local-only runtime options", str(ctx.exception))
        self.assertIn("right_hemisphere_backend=vjepa21", str(ctx.exception))

    def test_api_mode_without_local_features_is_valid(self) -> None:
        with patch(
            "calosum.bootstrap.infrastructure.settings._missing_local_dependency_stack",
            return_value=["torch", "peft", "transformers"],
        ):
            settings = InfrastructureSettings.from_sources(
                environ={
                    "CALOSUM_DEPENDENCY_MODE": "api",
                    "CALOSUM_RIGHT_BACKEND": "",
                    "CALOSUM_VECTOR_QUANTIZATION": "none",
                }
            )

        self.assertEqual(settings.dependency_mode, RuntimeDependencyMode.API)

    def test_mode_api_conflicts_with_dependency_local(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            InfrastructureSettings.from_sources(
                environ={
                    "CALOSUM_MODE": "api",
                    "CALOSUM_DEPENDENCY_MODE": "local",
                }
            )
        self.assertIn("CALOSUM_MODE=api conflicts", str(ctx.exception))

    def test_mode_local_conflicts_with_dependency_api(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            InfrastructureSettings.from_sources(
                environ={
                    "CALOSUM_MODE": "local",
                    "CALOSUM_DEPENDENCY_MODE": "api",
                }
            )
        self.assertIn("CALOSUM_MODE=local conflicts", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
