import pickle
from collections import defaultdict
from enum import Enum
from math import floor

import pandas as pd

PATH = 'datasets/'
ND = '-nd'


class Dataset(Enum):
    orders_train = 'orders_train'
    orders_class = 'orders_class'
    real_class = 'real_class'


class Column(Enum):
    order_id = 'orderID'
    order_date = 'orderDate'
    article_id = 'articleID'
    color_code = 'colorCode'
    size_code = 'sizeCode'
    product_group = 'productGroup'
    quantity = 'quantity'
    price = 'price'
    rrp = 'rrp'
    voucher_id = 'voucherID'
    voucher_amount = 'voucherAmount'
    customer_id = 'customerID'
    device_id = 'deviceID'
    payment_method = 'paymentMethod'
    return_quantity = 'returnQuantity'
    # new columns
    index = 'index'
    predicted_quantity = 'predictedQuantity'


class DataLoader(object):
    def __init__(self):
        self.path = PATH + '{name}{extension}.txt'

    def load_orders_train(self, as_ndarray):
        return self._load_data(Dataset.orders_train.value, as_ndarray)

    def load_orders_class(self, as_ndarray):
        return self._load_data(Dataset.orders_class.value, as_ndarray)

    def load_real_class(self):
        return self._load_data(Dataset.real_class.value, False)

    def _load_data(self, dataset_name, as_ndarray):
        if as_ndarray:
            with open(self.path.format_map(defaultdict(str, name=dataset_name, extension=ND)), 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            return data['data'], data['labels']
        else:
            return pd.read_csv(self.path.format_map(defaultdict(str, name=dataset_name)), sep=';')


class DataCleaner(object):
    def __init__(self, df):
        self.df = df

    def process_cleanup(self):
        self.cleanup_quantity()
        self.cleanup_price()
        self.cleanup_rrp()
        self.cleanup_size_code()
        self.cleanup_article_id()

    def cleanup_quantity(self):
        self.df = self.df[self.df['quantity'] > 0]
        self.df = self.df[self.df['quantity'] < 6]

    def cleanup_price(self, blocked_price_counts=lambda item_size: item_size < 100):
        self.df = self.df[self.df[Column.price.value] > 0]
        for blocked_price in self._apply_on_groups_items(Column.price.value, blocked_price_counts):
            self.df = self.df[self.df[Column.price.value] != blocked_price]

    def cleanup_rrp(self, blocked_rrp_counts=lambda item_size: item_size < 150):
        self.df = self.df[self.df[Column.rrp.value] > 0]
        for blocked_rrp in self._apply_on_groups_items(Column.rrp.value, blocked_rrp_counts):
            self.df = self.df[self.df[Column.rrp.value] != blocked_rrp]

    def cleanup_size_code(self, blocked_size_codes_counts=lambda item_size: item_size < 2000):
        for blocked_size_code in self._apply_on_groups_items(Column.size_code.value, blocked_size_codes_counts):
            self.df = self.df[self.df[Column.size_code.value] != blocked_size_code]

    def cleanup_article_id(self, blocked_article_id_counts=lambda item_size: item_size < 20):
        for blocked_article_id in self._apply_on_groups_items(Column.article_id.value, blocked_article_id_counts):
            self.df = self.df[self.df[Column.article_id.value] != blocked_article_id]

    def _apply_on_groups_items(self, column, condition):
        grouped_items = self.df[column].groupby(self.df[column]).size()
        for index, item in enumerate(grouped_items):
            if condition(item):
                yield grouped_items.index[index]


def preprocess_data(df_train, df_test):
    df_train, article_id_map = convert_items_to_numeric_values(df_train, Column.article_id.value)
    df_train, customer_id_map = convert_items_to_numeric_values(df_train, Column.customer_id.value)
    df_train, size_code_map = convert_items_to_numeric_values(df_train, Column.size_code.value)


def cleanup_data(dataframe):
    dataframe = compare_columns(dataframe, 'price', 'rrp', lambda x, y: x <= y)
    return dataframe


def filter_nasty_customers(dataframe, condition):
    grouped_items = dataframe['returnQuantity'].groupby(dataframe['customerID'])
    return show_most_unwanted_item(grouped_items, condition)


def filter_most_unwanted_colors(dataframe, condition):
    grouped_items = dataframe['returnQuantity'].groupby(dataframe['colorCode'])
    return show_most_unwanted_item(grouped_items, condition)


def compare_columns(dataframe, column1, column2, condition):
    return dataframe[condition(dataframe[column1], dataframe[column2])]


def show_most_unwanted_item(grouped_items, condition):
    nasty_items = []
    for name, group in grouped_items:
        returned_item = sum(group)
        if condition(returned_item):
            nasty_items.append(name)
    return nasty_items


def convert_items_to_numeric_values(dataframe, column, map_function):
    distinct_items_map = map_function(dataframe, column)
    dataframe[column] = dataframe[column].map(lambda column_value: distinct_items_map.get(column_value))
    return dataframe, distinct_items_map


def map_items_to_numeric_values(dataframe, column):
    distinct_items = dataframe[column].unique()
    converted_distinct_items = {value: key for key, value in enumerate(distinct_items)}
    return converted_distinct_items


def convert_dataframe2nparray(df, *columns):
    if len(columns) > 1:
        data = df.loc[:, columns]
        labels = df.iloc[:, df.shape[1] - 1]
    else:
        data = df.iloc[:, :df.shape[1] - 1]
        labels = df.iloc[:, df.shape[1] - 1]
    return data.values, labels.values


def export_dataframe_as_nparray(filename, df, *columns):
    data, labels = convert_dataframe2nparray(df, *columns)
    with open(PATH + filename, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)


def check_isnull(dataframe, columns):
    df = pd.isnull(dataframe)
    for column in columns:
        if len(df.groupby(column).size()) == 1:
            yield {column: False}
        else:
            yield {column: True}


def split_test_train(df, test_size):
    split_index = floor(df.shape[0] * test_size)
    df_train = df[split_index + 1:].copy()
    df_test = df[:split_index].copy()
    return df_train, df_test


if __name__ == '__main__':
    # data_df = load_orders_train()
    # data_df = preprocess_data(data_df)
    # for nullable in check_isnull(data_df,
    #                              ['articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID']):
    #     print(nullable)
    # export_dataframe_as_nparray(
    #     'orders_train',
    #     data_df,
    #     'articleID', 'colorCode', 'sizeCode', 'quantity', 'price', 'rrp', 'customerID'
    # )
    data_loader = DataLoader()
    df_train = data_loader.load_orders_train(as_ndarray=False)

    data_cleaner = DataCleaner(df_train)
    # data_cleaner.cleanup_quantity()
    # data_cleaner.cleanup_price()
    # data_cleaner.cleanup_rrp()
    # print(data_cleaner.df.head())
