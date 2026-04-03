import os
import unittest
from pathlib import Path

class TestDockerEnvAlignment(unittest.TestCase):
    def test_docker_compose_uses_right_model_path(self):
        docker_compose_path = Path("deploy/docker-compose.yml")
        self.assertTrue(docker_compose_path.exists())
        
        content = docker_compose_path.read_text()
        self.assertNotIn("CALOSUM_JEPA_MODEL_PATH", content, "Should have been replaced")
        self.assertIn("CALOSUM_RIGHT_MODEL_PATH", content, "Must use CALOSUM_RIGHT_MODEL_PATH")

if __name__ == "__main__":
    unittest.main()
