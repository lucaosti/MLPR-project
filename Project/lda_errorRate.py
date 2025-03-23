import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Carica i dati dal file di testo
data = np.loadtxt('normalized.txt', delimiter=",")

# Seleziona le prime 6 colonne (caratteristiche)
X = data[:, :-1]  # Caratteristiche
y = data[:, -1]   # Etichette (classe)

# Dividi il dataset in training e validation set (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Variabili per memorizzare l'error rate in funzione della dimensione di PCA
error_rates = []

# Testiamo diverse dimensioni per PCA (da 1 a 6 componenti principali)
for m in range(1, X_train.shape[1] + 1):
    # Applica PCA sul training set
    pca = PCA(n_components=m)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    # Applicare LDA per la classificazione
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train)

    # Previsioni sul set di validazione
    y_pred = lda.predict(X_val_pca)

    # Calcolare il tasso di errore
    error_rate = 1 - accuracy_score(y_val, y_pred)
    error_rates.append(error_rate)

# Visualizzare il tasso di errore in funzione del numero di componenti principali
plt.figure(figsize=(8, 6))
plt.plot(range(1, X_train.shape[1] + 1), error_rates, marker='o', linestyle='-', color='b')
plt.title('Tasso di errore in funzione del numero di componenti principali (PCA)')
plt.xlabel('Numero di componenti principali (m)')
plt.ylabel('Tasso di errore')
plt.grid(True)
plt.show()

# Stampare l'errore per ogni dimensione di PCA
for m, error_rate in zip(range(1, X_train.shape[1] + 1), error_rates):
    print(f"Numero di componenti principali: {m}, Tasso di errore: {error_rate:.4f}")