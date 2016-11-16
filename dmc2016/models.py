from dmc2016.datautils import Column, DataLoader


class HardPredictor(object):
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def predict(self):
        total_nasty_customers = 0
        for i, nasty_customers in enumerate(self._predict_nasty_customers(5)):
            self.df_test = self.df_test[self.df_test[Column.customer_id.value] != nasty_customers]
            total_nasty_customers = i

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


if __name__ == '__main__':
    dl = DataLoader()
    df = dl.load_orders_train(False)
    hp = HardPredictor(df, df)
    hp.predict()

#
#
# class SoftPredictor(object):
#     def __init__(self, df_train, df_test):
#         self.X_train = df_train.iloc[:, :df_train.shape[1] - 1]
#         self.y_train = df_train.iloc[:, df_train.shape[1] - 1]
#         self.X_test = df_test.iloc[:, :df_test.shape[1] - 1]
#         self.y_test = df_test.iloc[:, df_test.shape[1] - 1]
#
#     def fit(self):
#         pass
#
#     def predict(self):
#         pass
#
#     def evaluate(self, dmc_mode):
#         pass
#
#
# class LinearRegression(SoftPredictor):
#     def __init__(self, df_train, df_test):
#         super().__init__(df_train, df_test)
#         self.model = linear_model.LinearRegression()
#
#     def fit(self):
#         self.model.fit(self.X_train, self.y_train)
#
#     def predict(self):
#         y_pred = self.model.predict(self.X_test)
#         return y_pred
#
#     def evaluate(self, dmc_mode):
#         if dmc_mode:
#             print('dmc mode evaluation')
#             print(sum(abs(self.predict() - self.y_test)))
#         else:
#             pass
#
#
# def test(dataframe, column, map):
#     dataframe[column] = dataframe[column].map(lambda column_value: map.get(column_value))
#     return dataframe
#
#
# if __name__ == '__main__':
#     df_train = load_orders_train()
#     df_test = load_orders_class()
#     df_test_q = load_real_class()
#
#     frames = [df_test, df_test_q[Column.return_quantity.value]]
#     df_test = pd.concat(frames, axis=1)
#
#     df_train = df_train[
#         ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID', 'returnQuantity']]
#     df_test = df_test[
#         ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID', 'returnQuantity']]
#     df_test_q = df_test_q[Column.return_quantity.value]
#     df_train, article_id_map, customer_id_map, size_code_map = preprocess_data(df_train)
#     shuffled_df = df_train.iloc[np.random.permutation(len(df_train))]
#
#     df_test = test(df_test, 'articleID', article_id_map)
#     df_test = test(df_test, 'customerID', customer_id_map)
#     df_test = test(df_test, 'sizeCode', size_code_map)
#
#     shuffled_df = shuffled_df.dropna()
#     df_test = df_test.dropna()
#     print(df_test.shape)
#     # hard_predictor = HardPredictor(df_train, df_test)
#     # hard_predictor.predict()
#
#     linear_regression = LinearRegression(shuffled_df, df_test)
#     linear_regression.fit()
#     y_pred = linear_regression.predict()
#     print(sum(y_pred))
#     print(sum(abs(y_pred - df_test[Column.return_quantity.value])))
