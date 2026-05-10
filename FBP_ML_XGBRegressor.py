#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Daijingbo
# @Date  : 2019/5/27
# @Desc  :FBP ML
# http://www.captainbed.net/blog-acredjb
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer


DENSE_FEATURES = ['Oddset', 'li', 'bet365', 'interw', 'wl', 'w', 'ao']
SPARSE_FEATURES = ['10bet', 'jbb', 'ms', 'ysb', 'Pinnacle', 'SNAI']


def buildFeatures(data):
    # Old code used Imputer(axis=1) (per-row mean); SimpleImputer only supports
    # column-wise imputation, which is the more standard choice anyway.
    imputer = SimpleImputer(strategy='mean')
    sparse = imputer.fit_transform(data.loc[:, SPARSE_FEATURES])
    dense = data.loc[:, DENSE_FEATURES].to_numpy()
    return np.hstack([dense, sparse])


def featureSet(data):
    return buildFeatures(data), data['y'].values


def loadTestData(filePath):
    return buildFeatures(pd.read_csv(filePath))


def trainandTest(X_train, y_train, X_test, ids, output_path):
    model = xgb.XGBRegressor(
        max_depth=2,
        learning_rate=0.01,
        n_estimators=500,
        objective='reg:gamma',
    )
    model.fit(X_train, y_train)

    ans = model.predict(X_test)
    for prediction in ans:
        print(prediction)

    pd.DataFrame({'id': ids, 'y': ans}).to_csv(output_path, index=False)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='FBP XGBoost regressor')
    parser.add_argument('--train', default=os.path.join(here, 'FBP_train.csv'))
    parser.add_argument('--predict', default=os.path.join(here, 'FBP_predict.csv'))
    parser.add_argument('--output', default=os.path.join(here, 'FBP_submit.csv'))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    predict_df = pd.read_csv(args.predict)

    X_train, y_train = featureSet(train_df)
    X_test = buildFeatures(predict_df)
    ids = predict_df['id'].values

    trainandTest(X_train, y_train, X_test, ids, args.output)


if __name__ == '__main__':
    main()
