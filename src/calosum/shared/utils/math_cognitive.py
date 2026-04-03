from __future__ import annotations

import numpy as np
from typing import Any


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


def calculate_vfe(
    mu: np.ndarray,
    logvar: np.ndarray,
    prior_mu: np.ndarray,
    prior_logvar: np.ndarray,
    log_likelihood: float,
) -> float:
    """
    Calcula a Variational Free Energy (VFE).
    VFE = Complexity - Accuracy
    Complexity = DKL(Posterior || Prior)
    Accuracy = E_q[log p(o|s)] (aqui simplificado como log_likelihood direto)
    """
    complexity = kl_divergence_gaussian(mu, logvar, prior_mu, prior_logvar)
    accuracy = log_likelihood
    return complexity - accuracy


def calculate_efe(
    predicted_mu: np.ndarray,
    predicted_logvar: np.ndarray,
    preferred_mu: np.ndarray,
    preferred_logvar: np.ndarray,
    ambiguity: float = 0.0,
) -> float:
    """
    Calcula a Expected Free Energy (EFE).
    EFE = Risk + Ambiguity
    Risk = DKL(Predicted Posterior || Preferred Prior)
    Ambiguity = H(o|s) (incerteza residual esperada)
    """
    risk = kl_divergence_gaussian(predicted_mu, predicted_logvar, preferred_mu, preferred_logvar)
    return risk + ambiguity


def calculate_surprise(
    observation_latent: np.ndarray,
    predicted_mu: np.ndarray,
    predicted_logvar: np.ndarray,
) -> float:
    """
    Calcula o surprise score (neg-log-likelihood da observacao sob a predicao).
    Baseado na distancia de Mahalanobis simplificada para gaussiana diagonal.
    """
    var = np.exp(predicted_logvar) + 1e-6
    squared_diff = (observation_latent - predicted_mu) ** 2
    
    # Log-likelihood de uma Gaussiana: -0.5 * (log(2*pi*var) + (x-mu)^2 / var)
    # Ignorando constantes para o score de surpresa relativo
    surprise = 0.5 * np.sum(predicted_logvar + squared_diff / var)
    
    # Normalizacao sigmoidal simples para manter entre 0 e 1 se necessario
    return float(np.tanh(surprise / observation_latent.size))
