import numpy as np

def calculate_snr(original, noisy_or_denoised):
    """
    Calcule le Rapport Signal sur Bruit (SNR) pour l'audio en dB.
    """
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - noisy_or_denoised) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_psnr(original, processed):
    """
    Calcule le PSNR (Peak Signal-to-Noise Ratio) pour l'image en dB.
    Plus le PSNR est élevé, meilleure est la qualité de la restauration.
    """
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr