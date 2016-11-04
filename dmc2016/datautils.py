import pandas as pd


def load_orders_train(path='dmc2016/datasets'):
    return load_data(path + '/orders_train.txt')


def load_orders_class(path='dmc2016/datasets'):
    return load_data(path + '/orders_class.txt')


def load_data(file_name):
    return pd.read_csv(file_name, sep=';')


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
    dataframe.to_csv(filename)


def process_date(dataframe, filename='date.csv'):
    year, month, day = separate_date(dataframe)
    df = convert_date2dataframe(year, month, day)
    write_date2csv(filename, df)
