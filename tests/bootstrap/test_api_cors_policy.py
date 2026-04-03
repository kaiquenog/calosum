from __future__ import annotations

import os
import unittest

from calosum.bootstrap.entry.api import DEFAULT_ALLOWED_ORIGINS, resolve_cors_policy

class TestApiCorsPolicy(unittest.TestCase):
    def test_cors_policy_when_origins_provided(self):
        origins, allow_credentials = resolve_cors_policy({"CALOSUM_ALLOWED_ORIGINS": "http://my-ui.com"})
        self.assertEqual(origins, ["http://my-ui.com"])
        self.assertTrue(allow_credentials)

    def test_cors_policy_with_wildcard(self):
        origins, allow_credentials = resolve_cors_policy({"CALOSUM_ALLOWED_ORIGINS": "*"})
        self.assertEqual(origins, ["*"])
        self.assertFalse(allow_credentials)

    def test_cors_policy_defaults_to_local_ui_origins(self):
        origins, allow_credentials = resolve_cors_policy({})
        self.assertEqual(origins, list(DEFAULT_ALLOWED_ORIGINS))
        self.assertTrue(allow_credentials)

if __name__ == "__main__":
    unittest.main()
