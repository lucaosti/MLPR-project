import numpy as np
import matplotlib.pyplot as plt

# Load and normalize the data
def load_and_normalize_data(file_path):
    data = np.loadtxt(file_path, delimiter=",")
    X = data[:, :-1]
    y = data[:, -1]
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    X_normalized = (X - means) / stds
    return X_normalized, y

# Split the data into training and validation sets (80% training, 20% validation)
def train_test_split(X, y, test_size=0.2):
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Load and normalize the data
X, y = load_and_normalize_data('normalized.txt')

# Split the data into training and validation sets (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Compute the within-class scatter matrix (S_W)
def compute_within_class_scatter(X, y):
    classes = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    S_W = np.zeros((X.shape[1], X.shape[1]))
    
    for c in classes:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))
    
    return S_W

# Compute the between-class scatter matrix (S_B)
def compute_between_class_scatter(X, y):
    classes = np.unique(y)
    mean_overall = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    
    for c in classes:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)
        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(-1, 1)
        S_B += n_c * np.dot(mean_diff, mean_diff.T)
    
    return S_B

# Calculate the LDA projection direction w
S_W = compute_within_class_scatter(X_train, y_train)
S_B = compute_between_class_scatter(X_train, y_train)
S_W_inv = np.linalg.inv(S_W)
eigenvalues, eigenvectors = np.linalg.eig(np.dot(S_W_inv, S_B))

# Choose the eigenvector corresponding to the largest eigenvalue
w = np.real(eigenvectors[:, np.argmax(eigenvalues)])

# Project the training and test data onto w
X_proj_train = np.dot(X_train, w)
X_proj_test = np.dot(X_test, w)

# Calculate the means of the projections for each class
mean_class_0 = np.mean(X_proj_train[y_train == 0])
mean_class_1 = np.mean(X_proj_train[y_train == 1])

# Calculate the median of the projected values for each class
median_class_0 = np.median(X_proj_train[y_train == 0])
median_class_1 = np.median(X_proj_train[y_train == 1])

# **Threshold 1: Fixed threshold at 0.25**
threshold_1 = 0.25

# **Threshold 2: Fixed threshold at 0.75**
threshold_2 = 0.75

# **Threshold 3: Average of the projected class means (median method)**
threshold_3 = (median_class_0 + median_class_1) / 2

# List of thresholds for iteration
thresholds = [threshold_1, threshold_2, threshold_3]

# Loop through each threshold value and compute the error rate
for i, threshold in enumerate(thresholds, 1):
    # Make predictions on the test data
    y_pred = (X_proj_test > threshold).astype(int)
    
    # Calculate the error rate
    error_rate = np.mean(y_pred != y_test)
    
    # Print error rate for the current threshold
    print(f"Error rate for threshold {i} (Threshold = {threshold:.2f}): {error_rate * 100:.2f}%")

    # Visualize the results
    plt.figure(figsize=(8, 6))
    plt.hist(X_proj_train[y_train == 0], bins=30, alpha=0.7, label='Class 0', color='blue')
    plt.hist(X_proj_train[y_train == 1], bins=30, alpha=0.7, label='Class 1', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold: {threshold:.2f}')
    plt.title(f"LDA Projection with Threshold {i}")
    plt.xlabel("Projection Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Calculate the overlap percentage
    overlap = np.sum(np.minimum(np.histogram(X_proj_train[y_train == 0], bins=30)[0], np.histogram(X_proj_train[y_train == 1], bins=30)[0]))
    total = np.sum(np.histogram(X_proj_train[y_train == 0], bins=30)[0]) + np.sum(np.histogram(X_proj_train[y_train == 1], bins=30)[0])
    overlap_percentage = (overlap / total) * 100
    print(f"Overlap percentage for threshold {i}: {overlap_percentage:.2f}%")
