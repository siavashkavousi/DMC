import operator
import pickle
from collections import defaultdict
from enum import Enum
from functools import reduce
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

    def cleanup_quantity(self):
        self.df = self.df[self.df['quantity'] > 0]
        self.df = self.df[self.df['quantity'] < 6]

    def cleanup_price(self, allowed_price_counts=lambda item_size: item_size > 100):
        self.df = self.df[self.df['price'] > 0]
        self.df.reset_index(inplace=True)
        allowed_prices = self._group_item_index('price', allowed_price_counts)
        self.df = self.df.iloc[allowed_prices]

    def cleanup_rrp(self, allowed_rrp_counts=lambda item_size: item_size > 100):
        self.df = self.df[self.df['rrp'] > 0]
        self.df.reset_index(inplace=True, drop=True)
        allowed_rrps = group_item_index(self.df, 'rrp', allowed_rrp_counts)
        self.df = self.df.iloc[allowed_rrps]

    def _group_item_index(self, column, condition):
        grouped_items = self.df[column].groupby(self.df[column])
        allowed_items = []
        for name, group in grouped_items:
            indexes = [index for index in group.index]
            group_part_size = len(indexes)
            if condition(group_part_size):
                allowed_items.append(indexes)
        return reduce(operator.add, allowed_items)


def preprocess_data(df_train, df_test):
    df_train, article_id_map = convert_items_to_numeric_values(df_train, Column.article_id.value)
    df_train, customer_id_map = convert_items_to_numeric_values(df_train, Column.customer_id.value)
    df_train, size_code_map = convert_items_to_numeric_values(df_train, Column.size_code.value)


def cleanup_data(dataframe):
    dataframe = cleanup_quantity(dataframe)
    dataframe = cleanup_price(dataframe)
    dataframe = cleanup_rrp(dataframe)
    dataframe = cleanup_sizecode(dataframe)
    dataframe = cleanup_articleid(dataframe)
    dataframe = compare_columns(dataframe, 'price', 'rrp', lambda x, y: x <= y)
    return dataframe


def cleanup_quantity(dataframe):
    dataframe = dataframe[dataframe['quantity'] > 0]
    dataframe.reset_index(inplace=True, drop=True)
    dataframe = dataframe[dataframe['quantity'] < 6]
    return dataframe


def cleanup_price(dataframe, allowed_price_counts=lambda item_size: item_size > 100):
    dataframe = dataframe[dataframe['price'] > 0]
    dataframe.reset_index(inplace=True, drop=True)
    # group by price and remove prices with count < 100
    allowed_prices = group_item_index(dataframe, 'price', allowed_price_counts)
    dataframe = dataframe.iloc[allowed_prices]
    return dataframe


def cleanup_rrp(dataframe, allowed_rrp_counts=lambda item_size: item_size > 100):
    dataframe = dataframe[dataframe['rrp'] > 0]
    dataframe.reset_index(inplace=True, drop=True)
    allowed_rrps = group_item_index(dataframe, 'rrp', allowed_rrp_counts)
    dataframe = dataframe.iloc[allowed_rrps]
    return dataframe


def cleanup_sizecode(dataframe, allowed_sizecodes_counts=lambda item_size: item_size > 2000):
    dataframe.reset_index(inplace=True, drop=True)
    allowed_sizecodes = group_item_index(dataframe, 'sizeCode', allowed_sizecodes_counts)
    return dataframe.iloc[allowed_sizecodes]


def cleanup_articleid(dataframe, allowed_articleid_counts=lambda item_size: item_size > 20):
    dataframe.reset_index(inplace=True, drop=True)
    allowed_articleid = group_item_index(dataframe, 'articleID', allowed_articleid_counts)
    return dataframe.iloc[allowed_articleid]


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


def group_item_index(dataframe, column, condition):
    grouped_items = dataframe[column].groupby(dataframe[column])
    allowed_items = []
    for name, group in grouped_items:
        indexes = [index for index in group.index]
        group_part_size = len(indexes)
        if condition(group_part_size):
            allowed_items.append(indexes)
    return reduce(operator.add, allowed_items)


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
    data_cleaner.cleanup_quantity()
    data_cleaner.cleanup_price()
    data_cleaner.cleanup_rrp()
    print(data_cleaner.df)
