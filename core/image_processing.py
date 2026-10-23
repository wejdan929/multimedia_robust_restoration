import cv2
import numpy as np

def load_image(image_path, grayscale=True):
    """
    Charge une image depuis le disque. Par défaut en niveaux de gris.
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img

def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Ajoute un bruit de type 'sel et poivre' (pixels blancs et noirs aléatoires) 
    pour simuler des erreurs de transmission binaires.
    """
    noisy_image = image.copy()
    
    # Ajout du sel (pixels blancs)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[tuple(coords)] = 255
    
    # Ajout du poivre (pixels noirs)
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[tuple(coords)] = 0
    
    return noisy_image

def denoise_image(noisy_image, method='median', kernel_size=5):
    """
    Applique un filtre de débruitage (médian ou gaussien) pour restaurer l'image.
    Le filtre médian est particulièrement efficace contre le bruit sel et poivre.
    """
    if method == 'median':
        denoised = cv2.medianBlur(noisy_image, kernel_size)
    elif method == 'gaussian':
        denoised = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
    else:
        denoised = noisy_image
        
    return denoised

def save_image(image, output_path):
    """
    Sauvegarde l'image traitée sur le disque.
    """
    cv2.imwrite(output_path, image)