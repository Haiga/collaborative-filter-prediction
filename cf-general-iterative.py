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
X = np.array(X, dtype=np.float32)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

s = time.time()
dist = DistanceMetric.get_metric('euclidean')
num_features = X_train.shape[1]
batch_size = 100
k = 5
y_preds = []

for interval in np.array_split(np.arange(X_test.shape[0]), batch_size):
    all_distances = dist.pairwise(X_train[:, :], X_test[interval, :])

    topk_similar_instances = np.argpartition(-all_distances, -k, axis=0)[-k:]
    topk_similar_instances_labels = y_train[topk_similar_instances]

    modal_labels = stats.mode(topk_similar_instances_labels)
    y_pred = np.squeeze(modal_labels.mode)
    if y_pred.ndim == 0:
        y_preds.append([y_pred])
    else:
        y_preds.append(y_pred)

e = time.time()

y_pred = np.concatenate(y_preds)
s1 = time.time()
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
b_pred = clf.predict(X_test)
e1 = time.time()

print("My method: " + str(accuracy_score(y_test, y_pred)) + "\t time:" + str(e - s))
print("Baseline method: " + str(accuracy_score(y_test, b_pred)) + "\t time:" + str(e1 - s1))
end = True
