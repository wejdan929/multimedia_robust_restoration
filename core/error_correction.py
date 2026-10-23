import numpy as np

def encode_hamming_74(data_nibble):
    d = np.array(data_nibble, dtype=int) & 1
    # Calcul des bits de parité
    p1 = (d[0] + d[1] + d[2]) % 2
    p2 = (d[0] + d[1] + d[3]) % 2
    p3 = (d[0] + d[2] + d[3]) % 2
    return np.array([d[0], d[1], d[2], d[3], p1, p2, p3], dtype=int)

def decode_hamming_74(encoded_word):
    r = np.array(encoded_word, dtype=int) & 1
    
    # Matrice de contrôle de parité H (3 x 7) standard pour Hamming(7,4)
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 0, 0, 1]
    ], dtype=int)
    
    # Calcul du vecteur syndrome : s = H * r^T (mod 2)
    syndrome_vec = np.dot(H, r) % 2
    syndrome = int(syndrome_vec[0] + (syndrome_vec[1] << 1) + (syndrome_vec[2] << 2))
    
    corrected = r.copy()
    if syndrome != 0:
        # Recherche automatique de la colonne de H qui correspond au syndrome
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
