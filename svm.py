# Build a basic Support Vector Machine (SVM) model using both the original normalized features and features transformed by Principal Component Analysis (PCA)
# Retaining 1, 5, 10, and 30 principal components

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM model with original normalized features
svm_original = SVC(random_state=0)
svm_original.fit(X_train_scaled, y_train)
y_pred_original = svm_original.predict(X_test_scaled)
accuracy_original = accuracy_score(y_test, y_pred_original)
print("Accuracy with original normalized features:", accuracy_original)

# Define the number of principal components
components = [1, 5, 10, 30]
acc_pca = []

# Building SVM with PCA-transformed features
for i in components:
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm_pca = SVC(random_state=0)
    svm_pca.fit(X_train_pca, y_train)
    y_pred_pca = svm_pca.predict(X_test_pca)

    # Calculating accuracy depending on number of principal components
    acc_pca.append(accuracy_score(y_test, y_pred_pca))

print(f"Accuracy with 1 principal component (features transformed by PCA):", acc_pca[0])
print(f"Accuracy with 5 principal components (features transformed by PCA):", acc_pca[1])
print(f"Accuracy with 10 principal components (features transformed by PCA):", acc_pca[2])
print(f"Accuracy with 30 principal components (features transformed by PCA):", acc_pca[3])