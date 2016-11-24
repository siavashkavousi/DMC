import pandas as pd
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier

from dmc2016.data_utils import Column, DataLoader


class HardPredictor(object):
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test
        self.df_test_hard = pd.DataFrame()

    def predict(self):
        temp_dfs = []
        for nasty_customers in self._predict_nasty_customers(5):
            self.df_test.loc[self.df_test[Column.customer_id.value] == nasty_customers,
                             Column.return_quantity.value] = 1
            temp_dfs.append(self.df_test[self.df_test[Column.customer_id.value] == nasty_customers].copy())

        for nasty_customers in self._predict_nasty_customers(5):
            self.df_test = self.df_test[self.df_test[Column.customer_id.value] != nasty_customers]
        self.df_test_hard = pd.concat(temp_dfs)

        # for i, unwanted_colors in enumerate(self._predict_most_unwanted_colors(1)):
        #     self.df_test = self.df_test[self.df_test[Column.color_code.value] != unwanted_colors]
        #     temp = i
        # self.total += temp

    def _predict_nasty_customers(self, block_threshold):
        grouped_items = self.df_train[[Column.customer_id.value, Column.return_quantity.value]].groupby(
            self.df_train[Column.customer_id.value])
        for name, group in grouped_items:
            positive_return_percent = self._describe_item_group(group, Column.return_quantity.value)
            if positive_return_percent > block_threshold:
                yield name

    def _predict_most_unwanted_colors(self, block_threshold):
        grouped_items = self.df_train[[Column.color_code.value, Column.return_quantity.value]].groupby(
            self.df_train[Column.color_code.value])
        for name, group in grouped_items:
            positive_return_percent = self._describe_item_group(group, Column.return_quantity.value)
            if positive_return_percent > block_threshold:
                yield name

    @staticmethod
    def _describe_item_group(group, target_col):
        target_col_group_size = group.groupby(target_col).size()
        if len(target_col_group_size) > 1:
            try:
                positive_col_percent = sum(target_col_group_size[1:]) / float(target_col_group_size[0])
            except KeyError:
                # In case we don't have any zeros in target_col
                positive_col_percent = 1
        else:
            positive_col_percent = 0
        return positive_col_percent


class SoftPredictor(object):
    def __init__(self, df_train, df_test):
        self.X_train = df_train.iloc[:, :df_train.shape[1] - 1]
        self.y_train = df_train.iloc[:, df_train.shape[1] - 1]
        self.X_test = df_test.iloc[:, :df_test.shape[1] - 1]
        self.y_test = df_test.iloc[:, df_test.shape[1] - 1]

    def fit(self):
        pass

    def predict(self):
        pass

    def evaluate(self, dmc_mode):
        pass


class LinearRegression(SoftPredictor):
    def __init__(self, df_train, df_test):
        super().__init__(df_train, df_test)
        self.model = OneVsRestClassifier(linear_model.LinearRegression())

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.model.predict(self.X_test)
        return y_pred

    def evaluate(self, dmc_mode):
        if dmc_mode:
            print('dmc mode evaluation')
            print(sum(abs(self.predict() - self.y_test)))
        else:
            pass


if __name__ == '__main__':
    data_loader = DataLoader()
    df = data_loader.load_orders_train(False)

    df_train = df[680001:860000].copy()

    df_test = data_loader.load_orders_class(False)
    df_real_class = data_loader.load_real_class()
    df_test = pd.concat([df_test, df_real_class[Column.return_quantity.value]], axis=1)

    df_test = df[2000001:2200000].copy()

    df_train = df_train[
        ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID', 'returnQuantity']]
    df_test = df_test[
        ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID', 'returnQuantity']]

    # df_train = preprocess_data(df_train)
    # df_test = preprocess_data(df_test)

    print('df shapes init')
    print(df_train.shape)
    print(df_test.shape)

    hard_predictor = HardPredictor(df_train, df_test)
    hard_predictor.predict()
    df_test = hard_predictor.df_test
    total_hp = hard_predictor.total

    # print('df_test shape after hard prediction')
    # print(df_test.shape)
    # print('total hard prediction')
    # print(total_hp)
    #
    # linear_model = LinearRegression(df_train, df_test)
    # linear_model.fit()
    # y_pred = linear_model.predict()
    #
    # print(sum(abs(y_pred - df_test[Column.return_quantity.value])))
