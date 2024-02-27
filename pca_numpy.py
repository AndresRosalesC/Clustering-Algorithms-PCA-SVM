# Apply Principal Component Analysis (PCA) to the training set using the numpy package and keeping only one principal component.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Calculate covariance matrix, eigenvalues and eigenvectors for Principal Component Analysis (PCA)
cov_matrix = np.cov(X_train, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Keep only one principal component
selected_eigenvectors = eigenvectors[:, :1]

# Calculate the percentage of variance explained by the first component
t_var = np.sum(eigenvalues)
fcomp_var = eigenvalues[0]
percentage_variance_explained = (fcomp_var / t_var) * 100

print("Percentage of variance explained by the first Principal Component:", percentage_variance_explained)