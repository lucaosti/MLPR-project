import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Funzione per caricare e normalizzare i dati
def load_and_normalize_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_normalized = (X - means) / stds
    return X_normalized, y

# Caricare i dati normalizzati
X, y = load_and_normalize_data('normalized.txt')

# Suddividere i dati in training e validation set (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Variare il numero di dimensioni m per PCA e analizzare la performance
accuracies = []

for m in range(1, X_train.shape[1] + 1):  # Variare m da 1 a 6 (numero di caratteristiche)
    
    # Applicare PCA sui dati di addestramento (stima della PCA solo sui dati di addestramento)
    pca = PCA(n_components=m)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)  # Trasformare anche i dati di test
    
    # Applicare LDA sui dati trasformati tramite PCA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_pca, y_train)
    
    # Calcolare la precisione sul set di validazione
    accuracy = lda.score(X_test_pca, y_test)
    accuracies.append(accuracy)

    # Visualizzare i dati di addestramento e test proiettati su m componenti principali
    if m > 1:
        # Se m > 1, possiamo fare un grafico 2D
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train data')
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test data')
        plt.title(f'PCA with {m} components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.show()
    else:
        # Se m = 1, non possiamo fare un grafico 2D, quindi facciamo un grafico 1D
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train_pca, np.zeros_like(X_train_pca), c=y_train, cmap='coolwarm', marker='o', label='Train data')
        plt.scatter(X_test_pca, np.ones_like(X_test_pca), c=y_test, cmap='coolwarm', marker='x', label='Test data')
        plt.title(f'PCA with {m} component')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Class')
        plt.legend()
        plt.show()

# Grafico delle performance in funzione delle dimensioni di PCA
plt.plot(range(1, X_train.shape[1] + 1), accuracies, marker='o')
plt.title('Performance di LDA in funzione delle dimensioni di PCA')
plt.xlabel('Numero di componenti PCA (m)')
plt.ylabel('Accuratezza sul set di validazione')
plt.grid(True)
plt.show()

# Stampare il miglior risultato
best_m = np.argmax(accuracies) + 1
best_accuracy = np.max(accuracies)
print(f"Il miglior numero di componenti PCA è {best_m} con un'accuratezza di {best_accuracy * 100:.2f}%")