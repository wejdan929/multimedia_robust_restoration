# 📡 Multimedia Robustness Lab (R&D)

Plateforme R&D interactive développée en Python et Streamlit dédiée à l'analyse de robustesse, à la simulation de canaux de transmission bruités (Canal Binaire Symétrique - BSC) et au filtrage numérique pour les signaux multimédias (images et audio).

---

## 🚀 Fonctionnalités Principales

1. **Simulation de Canal & Restauration d'Image :**
   - Injection de bruit Sel & Poivre paramétrable.
   - Restauration par **Filtre Médian 2D** (`cv2.medianBlur`).
   - Évaluation quantitative des performances en direct via le **PSNR** (Peak Signal-to-Noise Ratio).
   - Support des images synthétiques (mires géométriques) et **importation de fichiers personnalisés** (PNG/JPG).

2. **Traitement & Filtrage Audio :**
   - Génération de signaux sinusoïdaux bruités (bruit additif gaussien) ou **importation de fichiers audio (WAV)**.
   - Filtrage par **Moyenne Mobile 1D**.
   - Calcul en direct du **SNR** (Signal-to-Noise Ratio) et visualisation comparative des signaux via Matplotlib.

3. **Codage Correcteur d'Erreurs (FEC) :**
   - Implémentation rigoureuse du **Code de Hamming(7,4)** basée sur la matrice de contrôle $H$ (`matrix dot product` et calcul de syndromes).
   - Simulation d'un **Canal Binaire Symétrique (BSC)** avec un Taux d'Erreur Binaire (BER) réglable.
   - Comparaison directe des flux binaires transmis *sans* codage vs *avec* correction d'erreurs.

---

## 🛠️ Stack Technique

- **Langage :** Python 3.10+
- **Interface Web :** Streamlit
- **Traitement & Calculs :** NumPy, SciPy, OpenCV
- **Visualisation :** Matplotlib
- **Tests Unitaires :** Pytest

---

## 📦 Installation & Lancement en Local

1. Clone le dépôt :
   ```bash
   git clone [https://github.com/wejdan929/multimedia_robust_restoration.git](https://github.com/wejdan929/multimedia_robust_restoration.git)
   cd multimedia_robust_restoration