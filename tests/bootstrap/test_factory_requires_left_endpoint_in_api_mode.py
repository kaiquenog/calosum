from __future__ import annotations

import unittest

from calosum.bootstrap.wiring.factory import CalosumAgentBuilder
from calosum.bootstrap.wiring.backend_resolvers import resolve_left_hemisphere
from calosum.bootstrap.infrastructure.settings import InfrastructureSettings, CalosumMode
from unittest.mock import patch

class TestFactoryRequiresLeftEndpointInApiMode(unittest.TestCase):
    def test_raises_runtime_error_when_missing_in_api_mode(self):
        settings = InfrastructureSettings(
            mode=CalosumMode.API,
            left_hemisphere_endpoint=None,
        )
        with self.assertRaisesRegex(RuntimeError, "CALOSUM_MODE=api requires CALOSUM_LEFT_ENDPOINT"):
            resolve_left_hemisphere(settings, "model-name")

    def test_passes_when_present_in_api_mode(self):
        settings = InfrastructureSettings(
            mode=CalosumMode.API,
            left_hemisphere_endpoint="http://example.com/api",
        )
        # Should not raise
        result, name = resolve_left_hemisphere(settings, "model-name")
        self.assertIsNotNone(result)

    def test_builder_fails_closed_in_api_mode_without_left_endpoint(self):
        builder = CalosumAgentBuilder(
            InfrastructureSettings(
                mode=CalosumMode.API,
                left_hemisphere_endpoint=None,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "CALOSUM_MODE=api requires CALOSUM_LEFT_ENDPOINT"):
            builder.build_left_hemisphere()

    def test_require_left_endpoint_env_var_blocks_provider_only_remote_backend(self):
        settings = InfrastructureSettings(
            mode=CalosumMode.LOCAL,
            left_hemisphere_backend="remote",
            left_hemisphere_provider="openai",
            left_hemisphere_endpoint=None,
        )
        with patch.dict("os.environ", {"CALOSUM_REQUIRE_LEFT_ENDPOINT": "true"}):
            with self.assertRaisesRegex(RuntimeError, "CALOSUM_REQUIRE_LEFT_ENDPOINT=1"):
                resolve_left_hemisphere(settings, "model-name")

if __name__ == "__main__":
    unittest.main()
