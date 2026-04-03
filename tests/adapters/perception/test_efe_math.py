from __future__ import annotations

import unittest

import numpy as np

from calosum.shared.utils.math_cognitive import calculate_efe, kl_divergence_gaussian


class EfeMathTests(unittest.TestCase):
    def test_kl_divergence_is_zero_for_identical_distributions(self) -> None:
        mu = np.array([0.1, -0.2, 0.3])
        logvar = np.array([-1.0, -1.0, -1.0])
        score = kl_divergence_gaussian(mu, logvar, mu, logvar)
        self.assertAlmostEqual(score, 0.0, places=6)

    def test_efe_increases_when_predicted_state_moves_away_from_preferred(self) -> None:
        preferred_mu = np.zeros(3)
        preferred_logvar = np.array([-2.0, -2.0, -2.0])
        near_mu = np.array([0.05, 0.0, -0.03])
        far_mu = np.array([1.0, -1.2, 0.9])
        logvar = np.array([-1.5, -1.5, -1.5])
        near_efe = calculate_efe(near_mu, logvar, preferred_mu, preferred_logvar, ambiguity=0.2)
        far_efe = calculate_efe(far_mu, logvar, preferred_mu, preferred_logvar, ambiguity=0.2)
        self.assertLess(near_efe, far_efe)


if __name__ == "__main__":
    unittest.main()
