from __future__ import annotations

import numpy as np

def surprise_from_predictive_error(
    observation_latent: np.ndarray,
    predicted_mu: np.ndarray,
    predicted_logvar: np.ndarray,
    saturate_max: float = 10.0,
) -> float:
    """
    Calcula o surprise score (neg-log-likelihood da observacao sob a predicao).
    Baseado na distancia de Mahalanobis simplificada para gaussiana diagonal.
    Evita logistic overflow e aplica saturacao numericamente estavel.
    """
    var = np.exp(predicted_logvar) + 1e-6
    squared_diff = (observation_latent - predicted_mu) ** 2
    
    # Log-likelihood de uma Gaussiana
    surprise = 0.5 * np.sum(predicted_logvar + squared_diff / var)
    normalized_surprise = surprise / max(1, observation_latent.size)
    
    # Saturação numericamente estável
    return float(np.clip(normalized_surprise, -saturate_max, saturate_max))

def calibrated_surprise_score(raw_surprise: float, history: list[float] | np.ndarray) -> float:
    """
    Calibrates a raw surprise score using z-score from history.
    """
    if len(history) < 5:
        # normalizacao sigmoidal simples para poucas amostras
        calibrated = float(np.tanh(raw_surprise))
        return float(max(0.0, min(1.0, (calibrated + 1.0) / 2.0)))
        
    hist_array = np.asarray(history, dtype=np.float32)
    mean = np.mean(hist_array)
    std = np.std(hist_array) + 1e-6
    z_score = (raw_surprise - mean) / std
    
    # Avoid overflow in exp
    z_score_clipped = np.clip(z_score - 1.0, -20.0, 20.0)
    calibrated = 1.0 / (1.0 + np.exp(-z_score_clipped))
    return float(round(max(0.0, min(1.0, calibrated)), 4))
