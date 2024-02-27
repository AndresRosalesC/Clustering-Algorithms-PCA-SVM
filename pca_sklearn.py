# Apply Principal Component Analysis (PCA) to the training set using sklearn package and keeping only one principal component

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create PCA model
pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)

# Obtain the percentage of variance explained by the first component
percentage_variance_explained = pca.explained_variance_ratio_[0] * 100

print("Percentage of variance explained by the first Principal Component:", percentage_variance_explained)