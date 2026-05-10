#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Daijingbo
# @Date  : 2019/6/16
# @Desc  :FBP ML XGBClassifier
# http://www.captainbed.net/blog-acredjb
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# `hg` exists in the training CSV but not in the prediction CSV, so it is
# excluded to keep the train/predict feature set aligned.
FEATURES = ['ysb', 'li', 'bet365', 'wl', 'ms', 'ao', 'interw', 'w', '10bet', 'SNAI']


def trainandTest(X, y, X_t, ids, output_path):
    # XGBoost expects class labels in [0, n_classes). The dataset uses {1, 2},
    # so encode before fitting and decode predictions back to the original space.
    classes = np.array(sorted(pd.unique(y)))
    y_encoded = np.searchsorted(classes, y)

    X_train, _, y_train, _ = train_test_split(
        X, y_encoded, test_size=0.25, random_state=33
    )

    vec = DictVectorizer(sparse=False)
    X_train_vec = vec.fit_transform(X_train.to_dict(orient='records'))
    X_t_vec = vec.transform(X_t.to_dict(orient='records'))

    model = XGBClassifier(eval_metric='mlogloss')
    model.fit(X_train_vec, y_train)

    ans = classes[model.predict(X_t_vec)]
    for prediction in ans:
        print(prediction)

    pd.DataFrame({'id': ids, 'y': ans}).to_csv(output_path, index=False)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='FBP XGBoost classifier')
    parser.add_argument('--train', default=os.path.join(here, 'FBP_train.csv'))
    parser.add_argument('--predict', default=os.path.join(here, 'FBP_predict.csv'))
    parser.add_argument('--output', default=os.path.join(here, 'FBP_submit.csv'))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    predict_df = pd.read_csv(args.predict)

    X_train = train_df[FEATURES]
    y_train = train_df['y']
    X_predict = predict_df[FEATURES]
    ids = predict_df['id'].values

    trainandTest(X_train, y_train, X_predict, ids, args.output)


if __name__ == '__main__':
    main()
