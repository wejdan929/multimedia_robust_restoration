import streamlit as st
import numpy as np
import cv2
import tempfile
import os

from core.audio_processing import add_gaussian_noise, denoise_audio_filter, save_audio
from core.image_processing import add_salt_and_pepper_noise, denoise_image
from core.metrics import calculate_snr, calculate_psnr

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Plateforme Multimédia Robuste",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Simulateur de Transmission & Débruitage Multimédia Robuste")
st.markdown("Plateforme locale d'évaluation de la robustesse des signaux (Image & Son) face au bruit et aux perturbations.")

# Menu latéral pour choisir entre Image et Audio
app_mode = st.sidebar.selectbox("Choisir le module", ["Traitement d'Image", "Traitement Audio"])

# ==========================================
# MODULE IMAGE
# ==========================================
if app_mode == "Traitement d'Image":
    st.header("🖼️ Module de Restauration d'Image")
    
    uploaded_file = st.file_uploader("Télécharger une image (PNG, JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Lecture de l'image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Originale")
            st.image(img_original, use_column_width=True, clamp=True)
            
        # Paramètres de bruit et filtre dans la barre latérale
        st.sidebar.subheader("Paramètres d'attaque & Filtre")
        noise_prob = st.sidebar.slider("Intensité Bruit Sel & Poivre", 0.0, 0.2, 0.05)
        kernel_size = st.sidebar.slider("Taille du noyau (Filtre Médian)", 3, 11, 5, step=2)
        
        # Application du bruit
        img_noisy = add_salt_and_pepper_noise(img_original, salt_prob=noise_prob, pepper_prob=noise_prob)
        
        with col2:
            st.subheader("2. Bruitée (Canal altéré)")
            st.image(img_noisy, use_column_width=True, clamp=True)
            
        # Débruitage
        img_denoised = denoise_image(img_noisy, method='median', kernel_size=kernel_size)
        psnr_val = calculate_psnr(img_original, img_denoised)
        
        with col3:
            st.subheader("3. Restaurée")
            st.image(img_denoised, use_column_width=True, clamp=True)
            st.success(f"**PSNR :** {psnr_val:.2f} dB")
            
    else:
        st.info("Veuillez importer une image pour commencer les tests de robustesse.")

# ==========================================
# MODULE AUDIO
# ==========================================
elif app_mode == "Traitement Audio":
    st.header("🎵 Module de Restauration Audio")
    
    uploaded_audio = st.file_uploader("Télécharger un fichier audio (.wav)", type=["wav"])
    
    if uploaded_audio is not None:
        import librosa
        
        # Sauvegarde temporaire pour lecture librosa
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        tfile.write(uploaded_audio.read())
        tfile.close()
        
        y_original, sr = librosa.load(tfile.name, sr=22050, mono=True)
        os.unlink(tfile.name)
        
        st.audio(y_original, format='audio/wav', sample_rate=sr)
        st.write("Signal audio original chargé avec succès.")
        
        # Paramètres audio dans la barre latérale
        st.sidebar.subheader("Paramètres Audio")
        noise_level = st.sidebar.slider("Niveau de Bruit Gaussien", 0.0, 0.1, 0.02)
        window_size = st.sidebar.slider("Taille de fenêtre (Filtre Moyenne Mobile)", 3, 21, 5, step=2)
        
        # Ajout de bruit et filtrage
        y_noisy = add_gaussian_noise(y_original, noise_level=noise_level)
        y_denoised = denoise_audio_filter(y_noisy, filter_type='moving_average', window_size=window_size)
        
        snr_val = calculate_snr(y_original, y_noisy)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Audio Bruité")
            st.audio(y_noisy, format='audio/wav', sample_rate=sr)
            st.warning(f"**SNR initial :** {snr_val:.2f} dB")
            
        with col_b:
            st.subheader("Audio Restauré")
            st.audio(y_denoised, format='audio/wav', sample_rate=sr)
            st.success("Filtrage appliqué avec succès !")
            
    else:
        st.info("Veuillez importer un fichier `.wav` pour tester le débruitage audio.")
        