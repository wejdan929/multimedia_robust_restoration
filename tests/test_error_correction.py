import numpy as np
from core.error_correction import encode_hamming_74, decode_hamming_74, binary_symmetric_channel

def test_hamming_perfect_transmission():
    original_nibble = [1, 0, 1, 1]
    encoded = encode_hamming_74(original_nibble)
    decoded, syndrome = decode_hamming_74(encoded)
    assert syndrome == 0
    np.testing.assert_array_equal(decoded, original_nibble)

def test_hamming_single_bit_correction():
    original_nibble = [1, 1, 0, 0]
    encoded = encode_hamming_74(original_nibble)
    corrupted = encoded.copy()
    corrupted[1] = 1 - corrupted[1]
    decoded, syndrome = decode_hamming_74(corrupted)
    assert syndrome != 0
    np.testing.assert_array_equal(decoded, original_nibble)

def test_binary_symmetric_channel():
    bits = np.zeros(1000, dtype=int)
    noisy = binary_symmetric_channel(bits, ber=0.1)
    error_rate = np.mean(bits != noisy)
    assert 0.05 < error_rate < 0.15
