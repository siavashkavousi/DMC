import pickle

import pandas as pd
# from keras.layers import Activation, Dropout
# from keras.layers import Dense
# from keras.models import Sequential
# from keras.optimizers import RMSprop, SGD
# from keras.utils import np_utils
# from sklearn.cross_validation import train_test_split

# from dmc2016.datautils import load_orders_train_data, PATH, load_orders_train


def export_model(filename, model):
    with open(PATH + 'models/' + filename, 'wb') as f:
        pickle.dump(model, f)


def import_model(filename):
    with open(PATH + 'models/' + filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def save_model(model):
    model.save_weights('weights.h5', overwrite=True)
    print('Saved weights into file')
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        f.write(model_json)
    print('Saved model structure to file')


def evaluate_model(y_true, y_pred):
    return sum(abs(y_true - y_pred))


# if __name__ == '__main__':
#     X, y_pred = load_orders_train_data()
#     X_train, X_test, y_train, y_test = train_test_split(X, y_pred, test_size=0.3)

    # # SVM classifier
    # print('training process')
    # clf = svm.SVC(decision_function_shape='ovo', verbose=True)
    # clf.fit(X_train, y_train)
    # print('saving the model')
    # export_model('svm', clf)
    # y_pred = clf.predict(X_train)
    #
    # print(accuracy_score(y_test, y_pred))

    # MLP classifier
    # nb_classes = 6
    #
    # Y_train = np_utils.to_categorical(y_train, nb_classes)
    # Y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    # model = Sequential()
    # model.add(Dense(128, input_dim=7, init='uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, init='uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes, init='uniform'))
    # model.add(Activation('softmax'))
    #
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # rmsprop = RMSprop(lr=0.1)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=rmsprop,
    #               metrics=['accuracy'])
    #
    # model.fit(X_train, Y_train,
    #           nb_epoch=1,
    #           batch_size=16)
    # score = model.evaluate(X_test, Y_test, batch_size=16)
    # y_pred = model.predict(X_test)
    # print(y_pred)
    # print(accuracy_score(y_test, y_pred))
    # print(evaluate_model(y_test, y_pred))
    # save_model(model)

    # Decision Tree Classifier
    # print('training process')
    # clf = tree.DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # print('predicting values')
    # y_pred = clf.predict(X_test)
    #
    # print(accuracy_score(y_test, y_pred))
    # print(evaluate_model(y_test, y_pred))
    #
    # test_data = load_orders_class()
    # test_data = test_data[['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID']]
    # y_pred = clf.predict(test_data)
    #
    # real_class = pd.read_csv('datasets/real_class.txt')
    # y_test = real_class['returnQuantity']
    #
    # print(accuracy_score(y_test, y_pred))
    # print(evaluate_model(y_test, y_pred))


class HardPredictor(object):
    def __init__(self, df):
        self.df = df
        self.predicted_df = pd.DataFrame()

    def predict_by_quantity(self, block_threshold):
        temp_df_list = []
        grouped_items = self.df[['quantity', 'returnQuantity']].groupby(self.df['quantity'])
        for name, group in grouped_items:
            print('quantity number: {name}'.format(name=name))
            return_quantity_group_size = group.groupby('returnQuantity').size()
            positive_return_percent = sum(return_quantity_group_size[1:]) / float(return_quantity_group_size[0])
            print('positive_return_percent: {value}'.format(value=positive_return_percent))

            if positive_return_percent < block_threshold:
                temp_df = pd.DataFrame([0 for _ in group.index], group.index)
                temp_df_list.append(temp_df)

        self.predicted_df = pd.concat(temp_df_list)




# if __name__ == '__main__':
#     df = load_orders_train()
#     predictor = HardPredictor(df)
#     predictor.predict_by_quantity(0.5)