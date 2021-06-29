import lightgbm
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import ndcg_score
import pandas as pd
# 0.512
from metrics import mNdcg

if __name__ == '__main__':
    path = r"D:\Colecoes\BD\MQ2007\MQ2007\Fold2\train.txt"
    X_train, y_train, qids_train = load_svmlight_file(path, query_id=True)
    path = r"D:\Colecoes\BD\MQ2007\MQ2007\Fold2\test.txt"
    X_test, y_test, qids_test = load_svmlight_file(path, query_id=True)
    path = r"D:\Colecoes\BD\MQ2007\MQ2007\Fold2\vali.txt"
    X_vali, y_vali, qids_vali = load_svmlight_file(path, query_id=True)

    X_train = X_train.toarray()
    X_test = X_test.toarray()
    X_vali = X_vali.toarray()

    qids_train_count = pd.DataFrame(qids_train).groupby(0)[0].count().to_numpy()
    qids_vali_count = pd.DataFrame(qids_vali).groupby(0)[0].count().to_numpy()
    qids_test_count = pd.DataFrame(qids_test).groupby(0)[0].count().to_numpy()

    model = lightgbm.LGBMRanker(
        objective="lambdarank",
        metric="ndcg"
    )

    model.fit(
        X=X_train,
        y=y_train,
        group=qids_train_count,
        eval_set=[(X_vali, y_vali)],
        eval_group=[qids_vali_count],
        eval_at=10,
        verbose=10
    )

    result = model.predict(X_test)
    summed_qid_count = 0
    grouped_preds_by_query = []
    grouped_rels_by_query = []
    for qid_count in qids_test_count:
        grouped_preds_by_query.append(result[summed_qid_count:summed_qid_count + qid_count])
        grouped_rels_by_query.append(y_test[summed_qid_count:summed_qid_count + qid_count])
        summed_qid_count += qid_count
    # print(mNdcg(grouped_rels_by_query, grouped_preds_by_query))
    print(np.mean(mNdcg(grouped_rels_by_query, grouped_preds_by_query, k=10, gains="exponential", no_relevant=False)))

    print('PyCharm')
