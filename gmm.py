# Apply the gaussian mixture model (GMM) to the test set directly. 
# Report the accuracy of the algorithm by comparing the results to the known labels.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings('ignore')

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply Gaussian Mixture Model (GMM) to the test dataset
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(X_test)

# Use GMM on the test dataset (predict labels)
gmm_labels = gmm.predict(X_test)

# Calculate accuracy scores for both cases (0 or 1)
accuracy1 = accuracy_score(y_test, gmm_labels)
accuracy2 = accuracy_score(y_test, 1 - gmm_labels)

# Report the higher accuracy score
accuracy = max(accuracy1, accuracy2)
print("Accuracy:", accuracy)