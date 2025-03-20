import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Carica i dati dal file di testo normalizzato
data = np.loadtxt('normalized.txt', delimiter=",")

# Seleziona le prime 6 colonne come caratteristiche
features = data[:, :6]
labels = data[:, 6]  # La settima colonna è la classe

# Lista di direzioni da usare per la PCA (come richiesto)
directions = [2, 0, 4, 1, 5, 3]  # Indici delle caratteristiche (3, 1, 5, 2, 6, 4 in zero-indexing)

# Applicare PCA per ciascuna delle direzioni specificate
for direction in directions:
    # Crea il modello PCA
    pca = PCA(n_components=1)  # Solo 1 componente per volta
    
    # Adatta il modello PCA alla singola colonna di interesse (usando solo una caratteristica per volta)
    pca_result = pca.fit_transform(features[:, [direction]])  # Usa solo la colonna indicata
    
    # Traccia l'istogramma dei dati proiettati
    plt.figure(figsize=(6, 4))
    plt.hist(pca_result, bins=30, edgecolor='black')
    plt.title(f'PCA Proiettato sulla Direzione {direction + 1}')
    plt.xlabel(f'Componente Principale {direction + 1}')
    plt.ylabel('Frequenza')
    plt.grid(True)
    plt.show()

# Crea scatter plot colorando per classe
for direction in directions:
    # Crea il modello PCA
    pca = PCA(n_components=1)  # Solo 1 componente per volta
    
    # Adatta il modello PCA alla singola colonna di interesse
    pca_result = pca.fit_transform(features[:, [direction]])  # Usa solo la colonna indicata
    
    # Crea il grafico scatter
    plt.figure(figsize=(8, 6))
    
    # Plot con colorazione per classe
    plt.scatter(pca_result, np.zeros_like(pca_result), c=labels, cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.title(f'PCA - Direzione {direction+1} (Colonna {direction+1})')
    plt.xlabel(f'PC {direction+1}')
    plt.ylabel('Proiezione')

    # Mostra il grafico
    plt.show()
