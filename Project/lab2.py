import numpy as np
import matplotlib.pyplot as plt

# Carica i dati dal file di testo
data = np.loadtxt('trainData.txt', delimiter=",")

# Seleziona le prime 6 colonne (caratteristiche)
first_6_columns = data[:, :6]

# Calcolare la media e la deviazione standard per ciascuna delle prime 6 colonne
means = first_6_columns.mean(axis=0)
stds = first_6_columns.std(axis=0)

# Normalizzare solo le prime 6 colonne (non la settima colonna)
first_6_columns_normalized = (first_6_columns - means) / stds

# Sostituire le prime 6 colonne normalizzate nei dati originali, mantenendo la settima colonna invariata
data[:, :6] = first_6_columns_normalized

# Salva il dataset normalizzato in un nuovo file chiamato normalized.txt
np.savetxt('normalized.txt', data, delimiter=',', fmt='%.6f')

# Carica i dati normalizzati
normalized = np.loadtxt('normalized.txt', delimiter=",")

# Seleziona solo le prime 6 colonne per il calcolo delle caratteristiche
data_features = normalized[:, :6]

# Seleziona l'ultima colonna che rappresenta le classi (0 o 1)
labels = normalized[:, -1]

# Funzione per calcolare la percentuale di overlap tra due classi per due caratteristiche
def calculate_overlap(class_0, class_1, threshold=0.1):
    overlap_count = np.sum(np.abs(class_0[:, np.newaxis] - class_1) < threshold)
    total_points = len(class_0) * len(class_1)
    overlap_percentage = (overlap_count / total_points) * 100
    return overlap_percentage

# **Prima figura: Istogrammi e scatter plot tra la prima e la seconda colonna**
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))  # 1 riga, 3 colonne
axes1 = axes1.ravel()

# Istogramma della prima colonna
axes1[0].hist(data_features[:, 0], bins=30, edgecolor='black')
axes1[0].set_title('Istogramma della prima colonna')
axes1[0].set_xlabel('Caratteristica 1')
axes1[0].set_ylabel('Frequenza')
axes1[0].grid(True)

# Istogramma della seconda colonna
axes1[1].hist(data_features[:, 1], bins=30, edgecolor='black')
axes1[1].set_title('Istogramma della seconda colonna')
axes1[1].set_xlabel('Caratteristica 2')
axes1[1].set_ylabel('Frequenza')
axes1[1].grid(True)

# Scatter plot tra la prima e la seconda colonna, colorato per classe
axes1[2].scatter(data_features[labels == 0, 0], data_features[labels == 0, 1], color='red', edgecolor='black', label='Classe 0', alpha=0.5)
axes1[2].scatter(data_features[labels == 1, 0], data_features[labels == 1, 1], color='blue', edgecolor='black', label='Classe 1', alpha=0.5)
axes1[2].set_xlabel('Caratteristica 1')
axes1[2].set_ylabel('Caratteristica 2')
axes1[2].set_title('Scatter plot tra Caratteristica 1 e Caratteristica 2')

# Calcolare l'overlap per la prima e seconda colonna
overlap_1_2 = calculate_overlap(data_features[labels == 0, 0], data_features[labels == 1, 0])
print(f"Percentuale di overlap tra Caratteristica 1 e Caratteristica 2: {overlap_1_2:.2f}%")

axes1[2].grid(True)
axes1[2].legend()

# Ottimizza la disposizione dei sottografici
plt.tight_layout()

# Mostra la prima figura
plt.show()

# **Seconda figura: Istogrammi e scatter plot tra la terza e la quarta colonna**
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))  # 1 riga, 3 colonne
axes2 = axes2.ravel()

# Istogramma della terza colonna
axes2[0].hist(data_features[:, 2], bins=30, edgecolor='black')
axes2[0].set_title('Istogramma della terza colonna')
axes2[0].set_xlabel('Caratteristica 3')
axes2[0].set_ylabel('Frequenza')
axes2[0].grid(True)

# Istogramma della quarta colonna
axes2[1].hist(data_features[:, 3], bins=30, edgecolor='black')
axes2[1].set_title('Istogramma della quarta colonna')
axes2[1].set_xlabel('Caratteristica 4')
axes2[1].set_ylabel('Frequenza')
axes2[1].grid(True)

# Scatter plot tra la terza e la quarta colonna, colorato per classe
axes2[2].scatter(data_features[labels == 0, 2], data_features[labels == 0, 3], color='red', edgecolor='black', label='Classe 0', alpha=0.5)
axes2[2].scatter(data_features[labels == 1, 2], data_features[labels == 1, 3], color='blue', edgecolor='black', label='Classe 1', alpha=0.5)
axes2[2].set_xlabel('Caratteristica 3')
axes2[2].set_ylabel('Caratteristica 4')
axes2[2].set_title('Scatter plot tra Caratteristica 3 e Caratteristica 4')

# Calcolare l'overlap per la terza e quarta colonna
overlap_3_4 = calculate_overlap(data_features[labels == 0, 2], data_features[labels == 1, 2])
print(f"Percentuale di overlap tra Caratteristica 3 e Caratteristica 4: {overlap_3_4:.2f}%")

axes2[2].grid(True)
axes2[2].legend()

# Ottimizza la disposizione dei sottografici
plt.tight_layout()

# Mostra la seconda figura
plt.show()

# **Terza figura: Istogrammi e scatter plot tra la quinta e la sesta colonna**
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))  # 1 riga, 3 colonne
axes3 = axes3.ravel()

# Istogramma della quinta colonna
axes3[0].hist(data_features[:, 4], bins=30, edgecolor='black')
axes3[0].set_title('Istogramma della quinta colonna')
axes3[0].set_xlabel('Caratteristica 5')
axes3[0].set_ylabel('Frequenza')
axes3[0].grid(True)

# Istogramma della sesta colonna
axes3[1].hist(data_features[:, 5], bins=30, edgecolor='black')
axes3[1].set_title('Istogramma della sesta colonna')
axes3[1].set_xlabel('Caratteristica 6')
axes3[1].set_ylabel('Frequenza')
axes3[1].grid(True)

# Scatter plot tra la quinta e la sesta colonna, colorato per classe
axes3[2].scatter(data_features[labels == 0, 4], data_features[labels == 0, 5], color='red', edgecolor='black', label='Classe 0', alpha=0.5)
axes3[2].scatter(data_features[labels == 1, 4], data_features[labels == 1, 5], color='blue', edgecolor='black', label='Classe 1', alpha=0.5)
axes3[2].set_xlabel('Caratteristica 5')
axes3[2].set_ylabel('Caratteristica 6')
axes3[2].set_title('Scatter plot tra Caratteristica 5 e Caratteristica 6')

# Calcolare l'overlap per la quinta e sesta colonna
overlap_5_6 = calculate_overlap(data_features[labels == 0, 4], data_features[labels == 1, 4])
print(f"Percentuale di overlap tra Caratteristica 5 e Caratteristica 6: {overlap_5_6:.2f}%")

axes3[2].grid(True)
axes3[2].legend()

# Ottimizza la disposizione dei sottografici
plt.tight_layout()

# Mostra la terza figura
plt.show()