# Apply k-means algorithm to the test set directly. 
# Report the accuracy of the algorithm by comparing the results to the known labels.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# Load the breast cancer dataset
data = load_breast_cancer()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.5, random_state=20)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply k-means algorithm to the test dataset
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_test)

# Calculate accuracy scores for both lcases (0 or 1)
accuracy1 = accuracy_score(y_test, kmeans.labels_)
accuracy2 = accuracy_score(y_test, 1 - kmeans.labels_)

# Report the correct accuracy score
accuracy = max(accuracy1, accuracy2)
print("Accuracy:", accuracy)