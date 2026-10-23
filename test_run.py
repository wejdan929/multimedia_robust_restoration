import os
import numpy as np
from core.audio_processing import load_audio, add_gaussian_noise, denoise_audio_filter, save_audio
from core.image_processing import load_image, add_salt_and_pepper_noise, denoise_image, save_image
from core.metrics import calculate_snr, calculate_psnr

print("=== VERIFICATION DES MODULES ===")

# 1. Test rapide Audio (génération d'un signal sinusoïdal factice pour tester sans fichier externe)
print("\n[1/2] Test du module Audio...")
sr = 22050
duration = 2  # secondes
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
y_original = 0.5 * np.sin(2 * np.pi * 440 * t) # Son La 440Hz

# Ajout de bruit et débruitage
y_noisy = add_gaussian_noise(y_original, noise_level=0.05)
y_denoised = denoise_audio_filter(y_noisy, filter_type='moving_average', window_size=5)

snr_val = calculate_snr(y_original, y_noisy)
print(f"-> Audio test OK ! SNR calculé : {snr_val:.2f} dB")

# 2. Test rapide Image (génération d'une image carrée grise factice)
print("\n[2/2] Test du module Image...")
img_original = np.ones((256, 256), dtype=np.uint8) * 128 # Image grise uniforme

# Ajout de bruit sel & poivre et débruitage
img_noisy = add_salt_and_pepper_noise(img_original, salt_prob=0.05, pepper_prob=0.05)
img_denoised = denoise_image(img_noisy, method='median', kernel_size=5)

psnr_val = calculate_psnr(img_original, img_denoised)
print(f"-> Image test OK ! PSNR de restauration : {psnr_val:.2f} dB")

print("\n=== TOUS LES TESTS DE BASE ONT REUSSI AVEC SUCCES ! ===")