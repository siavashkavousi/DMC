from math import sqrt

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

df = pd.read_excel('../datasets/gold_train.xlsx')

X, y = df.loc[:, ['store_num', 'daily_purchase', 'daily_sale', 'date']], df.loc[:, 'unit_price']
X, y = X['date'].values, y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# create model
model = Sequential()
model.add(Dense(512, input_dim=1, init='normal', activation='relu'))
model.add(Dense(256, init='normal', activation='relu'))
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dense(1, init='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, nb_epoch=5)
score = model.evaluate(X_test, y_test)
print('------------------')
print('MSE: %.2f' % score)
print('RMSE: %.2f' % sqrt(score))

