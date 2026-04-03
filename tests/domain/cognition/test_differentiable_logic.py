import numpy as np
from calosum.shared.utils.free_energy import expected_free_energy_refined, kl_divergence_gaussian

def test_kl_divergence_identical():
    mu_q = np.array([0.0, 0.0])
    logvar_q = np.array([0.0, 0.0])
    mu_p = np.array([0.0, 0.0])
    logvar_p = np.array([0.0, 0.0])
    kl = kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p)
    assert np.isclose(kl, 0.0)

def test_kl_divergence_difference():
    mu_q = np.array([1.0, 0.0])
    logvar_q = np.array([0.0, 0.0])
    mu_p = np.array([0.0, 0.0])
    logvar_p = np.array([0.0, 0.0])
    kl = kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p)
    assert kl > 0.0

def test_calculate_efe_refined():
    mu_q = np.array([1.0, 0.0])
    logvar_q = np.array([-1.0, -1.0]) # high certainty
    mu_p = np.array([0.0, 0.0])
    logvar_p = np.array([0.0, 0.0]) # low certainty prior
    
    # EFE = Complexity - Epistemic Value
    # Epistemic = epistemic_weight * (logvar_q - logvar_p).mean()
    # Epistemic here = 1.5 * (-1.0 - 0.0) = -1.5
    
    efe, terms = expected_free_energy_refined(mu_q, logvar_q, mu_p, logvar_p, epistemic_weight=1.5)
    
    # Complexity (KL) should be positive
    complexity = kl_divergence_gaussian(mu_q, logvar_q, mu_p, logvar_p)
    assert np.isclose(efe, complexity - (-1.5))
