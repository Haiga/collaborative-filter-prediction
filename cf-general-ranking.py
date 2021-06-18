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

from sklearn.datasets import load_svmlight_file


def get_data(t):
    # data = load_svmlight_file("D:\\Colecoes\\2003_td_dataset\\Fold1\\"+t+".txt", query_id=True)
    data = load_svmlight_file("D:\\Colecoes\\BD\\web10k-norm\\Fold1\\Norm." + t + ".txt", query_id=True)
    return data[0], data[1], data[2]


X, y, queries_id = get_data("train")


# iris = load_iris()
# iris = load_breast_cancer()
# iris = load_wine()


# X = iris.data

# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler.fit(X)
# X = scaler.transform(X)

X = power_transform(X.toarray())
print("transform")
# y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("split")
s = time.time()
dist = DistanceMetric.get_metric('euclidean')
all_distances = dist.pairwise(X_train, X_test)
print("pair")
k = 5
topk_similar_instances = np.argpartition(-all_distances, -k, axis=0)[-k:]
print("partition")
# y_train[topk_similar_instances]
# ranked_instances = np.argsort(all_distances, axis=0)
#
# l = np.argpartition(ranked_instances, -4, axis=0)[-4:]
# #for i in range(ranked_instances.shape[1]):
# #    all_distances[:, i] = all_distances[:, i][ranked_instances[:, i]]
topk_similar_instances_labels = y_train[topk_similar_instances]
modal_labels = stats.mode(topk_similar_instances_labels)
print("topk-modal")
y_pred = np.squeeze(modal_labels.mode)
e = time.time()
# conf = confusion_matrix(y_test, y_pred)
#
# df_cm = pd.DataFrame(conf, range(3), range(3))
# # plt.figure(figsize=(10,7))
# sns.set(font_scale=1.4)  # for label size
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
#
# plt.show()
s1 = time.time()
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
b_pred = clf.predict(X_test)
print("xgboost")
e1 = time.time()
print("My method: " + str(accuracy_score(y_test, y_pred)) + "\t time:" + str(e - s))
print("Baseline method: " + str(accuracy_score(y_test, b_pred)) + "\t time:" + str(e1 - s1))
end = True
