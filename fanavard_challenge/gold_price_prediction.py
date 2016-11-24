from math import sqrt

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

df = pd.read_excel('../datasets/gold_trade.xlsx')

df = df.rename(index=str, columns={u"روز": "day", u"ماه": "month", u"سال": "year",
                                   u"شماره فروشگاه": "store_num", u"تعداد خرید روزانه": "daily_purchase",
                                   u"تعداد فروش روزانه": "daily_sale",
                                   u"\u0642\u06cc\u0645\u062a \u062e\u0631\u06cc\u062f \u0647\u0631 \u0648\u0627\u062d\u062f ": "unit_price"})

# Drop unit costs with value '؟'
df = df[df['unit_price'] != u'؟']

X, y = df.iloc[:, :df.shape[1] - 1], df.iloc[:, df.shape[1] - 1]


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=1, init='normal', activation='relu'))
    model.add(Dense(256, init='normal', activation='relu'))
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def convert_to_date(df):
    temp_df = df[['day', 'month', 'year']].drop_duplicates()
    temp_df.reset_index(inplace=True, drop=True)
    mapped_date_values = temp_df.T.to_dict('list')
    for k, v in mapped_date_values.items():
        mapped_date_values[k] = '/'.join(map(str, v))
    mapped_date_values = {v: k for k, v in mapped_date_values.items()}

    def map_date(x):
        key = '/'.join([str(x['day']), str(x['month']), str(x['year'])])
        return mapped_date_values[key]

    df['date'] = df.apply(lambda x: map_date(x), axis=1)
    df.drop(['day', 'month', 'year'], axis=1, inplace=True)


convert_to_date(X)

X, y = X['date'].values, y.values

# enc = preprocessing.OneHotEncoder()
# enc.fit(X)
# print(enc.transform('4/4/2013').toarray())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=5, batch_size=5)

estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
print('real result')
print(y_test)
print('predicted result')
print(y_pred)
print('------------')
mse = mean_squared_error(y_test, y_pred)
print('RMSE')
print(sqrt(mse))

# kfold = KFold(n_splits=5, random_state=seed)
# results = cross_val_score(estimator, X, y, cv=kfold)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# print("RMSE: %.2f" % sqrt(results.mean()))

# X_real_test = np.array(['2013/4/13', '2013/7/13', '2013/7/14',
#                         '2013/11/7', '2013/12/15', '2014/2/9', '2014/2/17'])
#
# print(estimator.predict(X_real_test))
