from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class FreeEnergyTerms:
    epistemic_value: float
    instrumental_value: float
    complexity: float
    novelty_bonus: float

def kl_divergence_gaussian(
    mu1: np.ndarray, logvar1: np.ndarray, mu2: np.ndarray, logvar2: np.ndarray
) -> float:
    """
    Calcula a Divergencia KL entre duas distribuicoes Gaussianas diagonais.
    DKL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
    """
    var1 = np.exp(logvar1)
    var2 = np.exp(logvar2)
    
    # DKL = 0.5 * [ sum(var1/var2) + sum((mu2-mu1)^2 / var2) - d + sum(log(var2/var1)) ]
    term1 = var1 / var2
    term2 = (mu2 - mu1) ** 2 / var2
    term3 = logvar2 - logvar1
    
    dkl = 0.5 * np.sum(term1 + term2 - 1.0 + term3)
    return float(dkl)

def expected_free_energy_refined(
    posterior_mu: np.ndarray, 
    posterior_logvar: np.ndarray, 
    prior_mu: np.ndarray, 
    prior_logvar: np.ndarray, 
    epistemic_weight: float = 1.5,
    novelty_bonus: float = 0.0,
    policy_cost: float = 0.0
) -> tuple[float, FreeEnergyTerms]:
    """EFE rigorosa baseada na formulacao original de Friston"""
    # Instrumental value (pragmatic) + policy cost
    # Approximated by the complexity/risk term DKL(posterior || prior)
    complexity = kl_divergence_gaussian(posterior_mu, posterior_logvar, prior_mu, prior_logvar)
    instrumental_value = float(complexity) + float(policy_cost)
    
    # Epistemic value (novelty/ambiguity resolution)
    # Expected information gain
    if posterior_logvar.size > 0:
        epistemic_value = float(epistemic_weight * (posterior_logvar - prior_logvar).mean())
    else:
        epistemic_value = 0.0
        
    efe = float(instrumental_value - epistemic_value - novelty_bonus)
    
    terms = FreeEnergyTerms(
        epistemic_value=epistemic_value,
        instrumental_value=instrumental_value,
        complexity=float(complexity),
        novelty_bonus=novelty_bonus
    )
    return efe, terms

def variational_free_energy(
    mu: np.ndarray,
    logvar: np.ndarray,
    prior_mu: np.ndarray,
    prior_logvar: np.ndarray,
    reconstruction_error: float,
    uncertainty_regularizer: float = 0.01,
) -> float:
    """
    Calcula a Variational Free Energy (VFE).
    KL + reconstruction/prediction error + uncertainty regularizer
    """
    complexity = kl_divergence_gaussian(mu, logvar, prior_mu, prior_logvar)
    mean_uncertainty = float(np.mean(np.exp(logvar)))
    
    return float(complexity + reconstruction_error + (uncertainty_regularizer * mean_uncertainty))

def hierarchical_latent_prediction(
    coarse_latent: np.ndarray,
    fine_latent: np.ndarray,
    weight_coarse: float = 0.6,
) -> np.ndarray:
    """
    Predicao coarse-to-fine para vljepa.
    """
    # Simply interpolate or return fused representation for now
    if coarse_latent.shape != fine_latent.shape:
        # Resize or pad could be implemented here, but typically they are projected to same dim
        return fine_latent
    return coarse_latent * weight_coarse + fine_latent * (1.0 - weight_coarse)
