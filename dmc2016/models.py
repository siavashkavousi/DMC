import pickle

from sklearn import svm
from sklearn.cross_validation import train_test_split

from dmc2016.datautils import load_orders_train_data, PATH


# model = Sequential()
# model.add(Dense(64, input_dim=7, init='uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes, init='uniform'))
# model.add(Activation('softmax'))
#
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])
#
# model.fit(X_train, Y_train,
#           nb_epoch=20,
#           batch_size=16)
# score = model.evaluate(X_test, Y_test, batch_size=16)


def export_model(filename, model):
    with open(PATH + 'models/' + filename, 'wb') as f:
        pickle.dump(model, f)


def import_model(filename):
    with open(PATH + 'models/' + filename, 'rb') as f:
        return pickle.load(f, encoding='bytes')


if __name__ == '__main__':
    X, y = load_orders_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    print('training process')
    clf = svm.SVC(decision_function_shape='ovo', verbose=True)
    clf.fit(X_train, y_train)
    print('saving the model')
    export_model('svm', clf)
    print(clf.predict(X_train))
