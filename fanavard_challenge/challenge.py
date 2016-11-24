from math import sqrt

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def map_date(x, mapped_values):
    key = '/'.join([str(x['day']), str(x['month']), str(x['year'])])
    return mapped_values[key]


def convert_to_date(df):
    temp_df = df[['day', 'month', 'year']].drop_duplicates()
    temp_df.reset_index(inplace=True, drop=True)
    mapped_date_values = temp_df.T.to_dict('list')
    for k, v in mapped_date_values.items():
        mapped_date_values[k] = '/'.join(map(str, v))
    mapped_date_values = {v: k for k, v in mapped_date_values.items()}

    df['date'] = df.apply(lambda x: map_date(x, mapped_date_values), axis=1)
    df.drop(['day', 'month', 'year'], axis=1, inplace=True)
    return mapped_date_values


def inverse_map(map):
    return {v: k for k, v in map.items()}


# Set NAN data
def set_nan_unit_price_hard(x):
    if x['unit_price'] == '؟':
        x['unit_price'] = df.loc[df['date'] == x['date'], 'unit_price'].head(1).values[0]
    return x


if __name__ == '__main__':
    df = pd.read_excel('../datasets/gold_trade.xlsx')

    df = df.rename(index=str, columns={u"روز": "day", u"ماه": "month", u"سال": "year",
                                       u"شماره فروشگاه": "store_num", u"تعداد خرید روزانه": "daily_purchase",
                                       u"تعداد فروش روزانه": "daily_sale",
                                       u"\u0642\u06cc\u0645\u062a \u062e\u0631\u06cc\u062f \u0647\u0631 \u0648\u0627\u062d\u062f ": "unit_price"})

    mapped_date_values = convert_to_date(df)

    # Separate test set
    df_test = df.loc[df['date'].isin([mapped_date_values['13/4/2013'], mapped_date_values['13/7/2013'],
                                      mapped_date_values['14/7/2013'], mapped_date_values['7/11/2013'],
                                      mapped_date_values['15/12/2013'], mapped_date_values['9/2/2014'],
                                      mapped_date_values['17/2/2014']])].copy()
    df_test.drop('unit_price', axis=1, inplace=True)

    # Remove test set data
    df = df.loc[~df['date'].isin([mapped_date_values['13/4/2013'], mapped_date_values['13/7/2013'],
                                  mapped_date_values['14/7/2013'], mapped_date_values['7/11/2013'],
                                  mapped_date_values['15/12/2013'], mapped_date_values['9/2/2014'],
                                  mapped_date_values['17/2/2014']])]
    df_train = df.apply(set_nan_unit_price_hard, axis=1)

    X, y = df_train, df_train.pop('unit_price')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost classifier
    xgb.XGBClassifier()

    params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
              'objective': 'reg:linear', 'max_depth': 3, 'min_child_weight': 15,
              'eval_metric': 'rmse'}
    # XGBoost training and prediction process
    final_xgb = xgb.train(params, dtrain, num_boost_round=2990)

    y_pred = final_xgb.predict(dtest)
    print('y test data')
    print(y_test)
    print('y predicted data')
    print(y_pred)

    print('MSE: %.2f' % mean_squared_error(y_test, y_pred))
    print('RMSE: %.2f' % sqrt(mean_squared_error(y_test, y_pred)))

    # Unseen data prediction
    X_real_test = xgb.DMatrix(df_test)
    y_real_pred = final_xgb.predict(X_real_test)
    df_test['unit_price'] = y_real_pred

    inverse_mapped_date_values = inverse_map(mapped_date_values)
    df_test['date'] = df_test.apply(lambda x: inverse_mapped_date_values[x['date']], axis=1)

    writer = pd.ExcelWriter('../datasets/gold_test_pred.xlsx')
    df_test.to_excel(writer, 'Sheet1')
    writer.save()
