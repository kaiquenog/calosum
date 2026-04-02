from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from calosum.adapters.llm.llm_fusion import MultiSampleFusionLeftHemisphereAdapter
from calosum.bootstrap.infrastructure.settings import InfrastructureProfile, InfrastructureSettings
from calosum.bootstrap.wiring.backend_resolvers import _with_fusion_if_enabled


class _Provider:
    pass


class FusionResolverTests(unittest.TestCase):
    def test_ephemeral_profile_disables_fusion_by_default(self) -> None:
        settings = InfrastructureSettings(profile=InfrastructureProfile.EPHEMERAL)
        wrapped = _with_fusion_if_enabled(_Provider(), settings)
        self.assertFalse(isinstance(wrapped, MultiSampleFusionLeftHemisphereAdapter))

    def test_persistent_profile_enables_fusion_by_default(self) -> None:
        settings = InfrastructureSettings(profile=InfrastructureProfile.PERSISTENT)
        wrapped = _with_fusion_if_enabled(_Provider(), settings)
        self.assertTrue(isinstance(wrapped, MultiSampleFusionLeftHemisphereAdapter))

    def test_env_flag_can_disable_fusion(self) -> None:
        settings = InfrastructureSettings(profile=InfrastructureProfile.PERSISTENT)
        with patch.dict(os.environ, {"CALOSUM_FUSION_ENABLED": "false"}):
            wrapped = _with_fusion_if_enabled(_Provider(), settings)
        self.assertFalse(isinstance(wrapped, MultiSampleFusionLeftHemisphereAdapter))
