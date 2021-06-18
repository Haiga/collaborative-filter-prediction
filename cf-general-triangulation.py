from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import DistanceMetric
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler, power_transform
import time

# iris = load_iris()
iris = load_breast_cancer()
# iris = load_wine()

X = iris.data
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler.fit(X)
# X = scaler.transform(X)
X = power_transform(X)

# num_locals = X.shape[1] + 1
num_features = X.shape[1]
# num_locals = int(num_features * 0.01) + 1
num_locals = 31
X_locals = (np.random.rand(num_locals, num_features) - 0.5) * 2

y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

s = time.time()
dist = DistanceMetric.get_metric('euclidean')
all_distances_train = dist.pairwise(X_train, X_locals)
all_distances_test = dist.pairwise(X_test, X_locals)
all_distances = dist.pairwise(all_distances_train, all_distances_test)
k = 1
topk_similar_instances = np.argpartition(-all_distances, -k, axis=0)[-k:]
topk_similar_instances_labels = y_train[topk_similar_instances]
modal_labels = stats.mode(topk_similar_instances_labels)
y_pred = np.squeeze(modal_labels.mode)
e = time.time()

s1 = time.time()
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
b_pred = clf.predict(X_test)
e1 = time.time()

print("My method: " + str(accuracy_score(y_test, y_pred)) + "\t time:" + str(e - s))
print("Baseline method: " + str(accuracy_score(y_test, b_pred)) + "\t time:" + str(e1 - s1))

end = True
