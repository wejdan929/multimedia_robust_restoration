import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- MODULES EMBARQUÉS (Zéro erreur d'importation) ---

def median_filter_2d(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def moving_average_1d(signal, window_size=5):
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')

def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_snr(signal_clean, signal_noisy):
    noise = signal_noisy - signal_clean
    signal_power = np.mean(signal_clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)

def encode_hamming_74(data_nibble):
    d = np.array(data_nibble, dtype=int) & 1
    p1 = (d[0] + d[1] + d[2]) % 2
    p2 = (d[0] + d[1] + d[3]) % 2
    p3 = (d[0] + d[2] + d[3]) % 2
    return np.array([d[0], d[1], d[2], d[3], p1, p2, p3], dtype=int)

def decode_hamming_74(encoded_word):
    r = np.array(encoded_word, dtype=int) & 1
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1]
    ], dtype=int)
    syndrome_vec = np.dot(H, r) % 2
    syndrome = int(syndrome_vec[0] + (syndrome_vec[1] << 1) + (syndrome_vec[2] << 2))
    
    corrected = r.copy()
    if syndrome != 0:
        for i in range(7):
            if np.array_equal(H[:, i], syndrome_vec):
                corrected[i] = 1 - corrected[i]
                break
    return corrected[:4], syndrome

def binary_symmetric_channel(bit_stream, ber=0.01):
    noisy_stream = bit_stream.copy()
    mask = np.random.rand(len(bit_stream)) < ber
    noisy_stream[mask] = 1 - noisy_stream[mask]
    return noisy_stream

# --- INTERFACE STREAMLIT ---

st.set_page_config(page_title="Multimedia Robustness Lab", layout="wide")

st.title("📡 Labo de Transmission & Restauration Multimédia")
st.caption("Plateforme R&D d'analyse de robustesse, simulation de canal (BSC/Hamming) et filtrage numérique")

tab1, tab2, tab3 = st.tabs(["🖼️ Image & Canal Radio", "🎵 Audio & Filtres", "🧪 Canal Binaire & Hamming (FEC)"])

with tab1:
    st.header("Simulation de Canal & Restauration Image")
    col1, col2 = st.columns(2)
    
    with col1:
        noise_level = st.slider("Densité de bruit Sel & Poivre", 0.0, 0.3, 0.05, 0.01)
        kernel_size = st.select_slider("Taille du noyau (Filtre Médian)", options=[3, 5, 7], value=3)
        
        img_size = 256
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (220, 220), 255, -1)
        cv2.circle(img, (128, 128), 60, 0, -1)
        cv2.line(img, (0, 0), (256, 256), 127, 5)
        
        noisy_img = img.copy()
        num_noise = int(noise_level * img.size)
        coords_salt = [np.random.randint(0, i - 1, int(num_noise / 2)) for i in img.shape]
        noisy_img[tuple(coords_salt)] = 255
        coords_pepper = [np.random.randint(0, i - 1, int(num_noise / 2)) for i in img.shape]
        noisy_img[tuple(coords_pepper)] = 0
        
        restored_img = median_filter_2d(noisy_img, kernel_size=kernel_size)
        
        psnr_noisy = calculate_psnr(img, noisy_img)
        psnr_restored = calculate_psnr(img, restored_img)
        
        st.metric("PSNR Image Bruitée", f"{psnr_noisy:.2f} dB")
        st.metric("PSNR Image Restaurée", f"{psnr_restored:.2f} dB", delta=f"{psnr_restored - psnr_noisy:+.2f} dB")

    with col2:
        st.image(noisy_img, caption="Image après Canal Bruité", use_container_width=True)
        st.image(restored_img, caption="Image après Filtrage Médian", use_container_width=True)

with tab2:
    st.header("Traitement & Filtrage Audio")
    audio_noise = st.slider("Amplitude du Bruit Audio", 0.0, 0.5, 0.1, 0.02)
    window_size = st.slider("Taille de Fenêtre (Moyenne Mobile)", 3, 21, 5, step=2)
    
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    noise = np.random.normal(0, audio_noise, len(t))
    noisy_signal = clean_signal + noise
    restored_signal = moving_average_1d(noisy_signal, window_size=window_size)
    
    snr_noisy = calculate_snr(clean_signal, noisy_signal)
    snr_restored = calculate_snr(clean_signal, restored_signal)
    
    col_a, col_b = st.columns(2)
    col_a.metric("SNR Signal Bruité", f"{snr_noisy:.2f} dB")
    col_b.metric("SNR Signal Restauré", f"{snr_restored:.2f} dB", delta=f"{snr_restored - snr_noisy:+.2f} dB")
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t[:200], clean_signal[:200], label="Original", alpha=0.7)
    ax.plot(t[:200], noisy_signal[:200], label="Bruité", alpha=0.5)
    ax.plot(t[:200], restored_signal[:200], label="Restauré", color="red")
    ax.legend()
    st.pyplot(fig)

with tab3:
    st.header("Codage Correcteur d'Erreurs : Hamming(7,4) vs Canal BSC")
    st.write("Démonstration de transmission binaire avec et sans correction d'erreurs.")
    
    ber = st.slider("Taux d'Erreur Binaire du Canal (BER)", 0.0, 0.2, 0.03, 0.01)
    
    data = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
    data_np = np.array(data)
    
    direct_rx = binary_symmetric_channel(data_np, ber=ber)
    direct_errors = np.sum(data_np != direct_rx)
    
    encoded_stream = []
    for i in range(0, len(data), 4):
        nibble = data[i:i+4]
        encoded_stream.extend(encode_hamming_74(nibble))
        
    encoded_np = np.array(encoded_stream)
    rx_encoded = binary_symmetric_channel(encoded_np, ber=ber)
    
    decoded_stream = []
    total_syndromes = 0
    for i in range(0, len(rx_encoded), 7):
        word = rx_encoded[i:i+7]
        decoded_nibble, syn = decode_hamming_74(word)
        decoded_stream.extend(decoded_nibble)
        if syn != 0:
            total_syndromes += 1
            
    decoded_np = np.array(decoded_stream)
    fec_errors = np.sum(data_np != decoded_np)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Erreurs SANS Codage", f"{direct_errors} bits")
    m2.metric("Erreurs AVEC Hamming(7,4)", f"{fec_errors} bits", delta=f"{fec_errors - direct_errors:d} bits")
    m3.metric("Anomalies Détectées (Syndromes)", f"{total_syndromes} blocs")
    
    st.subheader("Comparaison des Flux Binaires")
    st.text(f"Original : {data_np.tolist()}")
    st.text(f"Sans FEC  : {direct_rx.tolist()}")
    st.text(f"Avec FEC : {decoded_np.tolist()}")