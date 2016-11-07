import operator
from functools import reduce

import pandas as pd

PATH = 'datasets/'


def load_orders_train():
    return load_data('orders_train.txt')


def load_orders_class():
    return load_data('orders_class.txt')


def load_data(file_name):
    return pd.read_csv(PATH + file_name, sep=';')


def separate_date(dataframe):
    date = dataframe['orderDate'].str.split('-')
    year = [i[0] for i in date]
    month = [i[1] for i in date]
    day = [i[2] for i in date]
    return year, month, day


def convert_date2dataframe(year, month, day):
    d = {'year': year, 'month': month, 'day': day}
    return pd.DataFrame(d)


def write_date2csv(filename, dataframe):
    dataframe.to_csv(filename, index=False)


def process_date(dataframe, filename='date.csv'):
    year, month, day = separate_date(dataframe)
    df = convert_date2dataframe(year, month, day)
    write_date2csv(PATH + filename, df)


def preprocess_data(dataframe):
    cleanup_data(dataframe)
    dataframe = convert_items_content(dataframe, 'articleID')
    dataframe = convert_items_content(dataframe, 'customerID')
    dataframe = convert_items_content(dataframe, 'sizeCode')
    return dataframe


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


def convert_items_content(dataframe, column):
    distinct_items_map = map_items_content(dataframe, column)
    dataframe[column] = dataframe[column].map(lambda column_value: distinct_items_map.get(column_value))
    return dataframe


def map_items_content(dataframe, column):
    grouped_items = dataframe[column].groupby(dataframe[column])
    distinct_items = [name for name, _ in grouped_items]
    map_distinct_items = {value: key for key, value in enumerate(distinct_items)}
    return map_distinct_items


if __name__ == '__main__':
    PATH = 'datasets/'
    df = load_orders_train()
    print(preprocess_data(df))
