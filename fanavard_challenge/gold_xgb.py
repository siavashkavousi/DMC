from math import sqrt

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def load_data():
    df = pd.read_excel('../datasets/gold_train.xlsx')
    X, y = df, df.pop('unit_price')
    return X.values, y.values


def load_test_data():
    df_test = pd.read_excel('../datasets/gold_test.xlsx')
    return xgb.DMatrix(df_test.values)


if __name__ == '__main__':
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost classifier
    xgb.XGBClassifier()

    # XGBoost native cv
    # params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
    #           'objective': 'reg:linear', 'max_depth': 3, 'min_child_weight': 30,
    #           'eval_metric': 'rmse'}
    # xgb_cv = xgb.cv(params=params, dtrain=dtrain, num_boost_round=3000, nfold=10, early_stopping_rounds=100)
    # print(xgb_cv.tail(10))

    # XGBoost training and prediction process
    # final_xgb = xgb.train(params, dtrain, num_boost_round=2995)
    #
    # print(dtest)
    # y_pred = final_xgb.predict(dtest)
    #
    # print(y_test)
    # print(y_pred)

    # Sklearn cv process
    cv_params = {'min_child_weight': [15, 18, 20], 'learning_rate': [0.2, 0.1, 0.01],
                 'subsample': [0.7, 0.8, 0.9]}
    ind_params = {'max_depth': 3, 'n_estimators': 1000, 'seed': 0,
                  'colsample_bytree': 0.8, 'objective': 'reg:linear'}

    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), cv_params, cv=10, verbose=1)

    optimized_GBM.fit(X_train, y_train)

    print(optimized_GBM.grid_scores_)

    y_pred = optimized_GBM.predict(X_test)
    print(y_pred)

    print('MSE: %.2f' % mean_squared_error(y_test, y_pred))
    print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, y_pred)))

    # Unseen data prediction
    # df_test = pd.read_excel('../datasets/gold_test.xlsx')
    # X_real_test = xgb.DMatrix(df_test.values)
    # y_real_pred = final_xgb.predict(X_real_test)
    # print(y_real_pred)
