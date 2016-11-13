import numpy as np

from dmc2016.datautils import load_orders_train, Column, split_test_train


class HardPredictor(object):
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def _predict_zero_for_quantity(self, block_threshold):
        def describe_quantity_group(group):
            return_quantity_group_size = group.groupby(Column.return_quantity.value).size()
            if len(return_quantity_group_size) > 1:
                positive_return_percent = sum(return_quantity_group_size[1:]) / float(return_quantity_group_size[0])
            else:
                positive_return_percent = 0
            return positive_return_percent

        grouped_items = self.df_train[[Column.quantity.value, Column.return_quantity.value]].groupby(
            self.df_train[Column.quantity.value])
        for name, group in grouped_items:
            positive_return_percent = describe_quantity_group(group)
            if positive_return_percent < block_threshold:
                yield name

    def predict(self):
        for quantity in self._predict_zero_for_quantity(0.5):
            self.df_test.loc[self.df_test.quantity == quantity, Column.predicted_quantity.value] = 0


class SoftPredictor(object):
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test


class LinearRegression(SoftPredictor):
    def __init__(self, df_train, df_test):
        super().__init__(df_train, df_test)

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    df = load_orders_train()
    shuffled_df = df.iloc[np.random.permutation(len(df))]
    df_train, df_test = split_test_train(shuffled_df, 0.3)
    predictor = HardPredictor(df_train, df_test)
    predictor.predict()
