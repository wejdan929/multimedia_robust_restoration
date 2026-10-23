import numpy as np
import librosa
import soundfile as sf

def load_audio(file_path, sr=22050):
    """
    Charge un fichier audio et le convertit en mono à une fréquence d'échantillonnage fixe.
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def add_gaussian_noise(y, noise_level=0.005):
    """
    Ajoute un bruit blanc gaussien au signal audio pour simuler des perturbations.
    """
    noise = np.random.normal(0, 1, len(y))
    y_noisy = y + noise_level * noise
    return y_noisy

def denoise_audio_filter(y_noisy, filter_type='moving_average', window_size=5):
    """
    Applique un filtre simple (moyenne mobile) pour atténuer le bruit haute fréquence.
    """
    if filter_type == 'moving_average':
        window = np.ones(window_size) / window_size
        y_denoised = np.convolve(y_noisy, window, mode='same')
        return y_denoised
    return y_noisy

def save_audio(y, sr, output_path):
    """
    Sauvegarde le signal audio traité dans un fichier .wav
    """
    sf.write(output_path, y, sr)