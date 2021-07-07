import lightgbm
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
import pandas as pd
from metrics import mNdcg
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt

def read_collection(base_path, fold, role):
    sfold = str(fold)
    return load_svmlight_file(f"{base_path}/Fold{sfold}/{role}.txt", query_id=True)


if __name__ == '__main__':
    base_path = r"D:\Colecoes\BD\MQ2007\MQ2007"
    fold = 2

    X_train, y_train, qids_train = read_collection(base_path, fold, "train")
    X_test, y_test, qids_test = read_collection(base_path, fold, "test")
    X_vali, y_vali, qids_vali = read_collection(base_path, fold, "vali")

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    X_vali = X_vali.toarray()

    qids_train_count = pd.DataFrame(qids_train).groupby(0)[0].count().to_numpy()
    qids_vali_count = pd.DataFrame(qids_vali).groupby(0)[0].count().to_numpy()
    qids_test_count = pd.DataFrame(qids_test).groupby(0)[0].count().to_numpy()

    pd_vali = pd.DataFrame(X_vali)
    pd_vali['qid'] = qids_vali
    pd_vali['y'] = y_vali

    cols = range(46)
    # one_query = pd_vali.query("qid == 7968")
    # one_query = pd_vali.query("qid == 8137")
    # one_query = pd_vali.query("qid == 8370")
    # one_query = pd_vali.query("qid == 8946")
    # one_query = pd_vali.query("qid == 9180")
    us = []
    for qid in np.unique(pd_vali.qid):
        one_query = pd_vali.query("qid == "+str(qid))
        u = []
        for i in cols:
            corr, _ = pearsonr(one_query.y, one_query[i])
            corr1, _ = spearmanr(one_query.y, one_query[i])
            corr2, _ = kendalltau(one_query.y, one_query[i])
            u.append([corr, corr1, corr2])
        us.append(u)
    us = np.array(us)
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # ax.plot(cols, np.array(u)[:, 0], 'ok', label='pearson')
    # ax.plot(cols, np.array(u)[:, 1], 'ob', label='spearman')
    # ax.plot(cols, np.array(u)[:, 2], 'or', label='kendall')
    # # plt.legend()
    # # lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, 0))
    # lgd = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    # plt.tight_layout()
    # plt.show()

    # pd_p = pd.DataFrame(us[:, :, 0])
    # pd_p["corr"] = "pearson"
    # pd_s = pd.DataFrame(us[:, :, 1])
    # pd_s["corr"] = "spearman"
    # pd_k = pd.DataFrame(us[:, :, 2])
    # pd_k["corr"] = "kendall"
    #
    #
    # pd_all = pd_p.copy()
    # pd_all.append([pd_s, pd_k])
    # plt.boxplot(us[:, :, 0])
    # plt.show()
    plt.boxplot(np.nan_to_num(us[:, :, 1]))
    plt.show()
    k = 0
