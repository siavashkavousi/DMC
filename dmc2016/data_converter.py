import pandas as pd

from dmc2016.datautils import PATH


def process_date(dataframe, filename='date.csv'):
    def separate_date():
        date = dataframe['orderDate'].str.split('-')
        year = [i[0] for i in date]
        month = [i[1] for i in date]
        day = [i[2] for i in date]
        return year, month, day

    def convert_date2dataframe(year, month, day):
        d = {'year': year, 'month': month, 'day': day}
        return pd.DataFrame(d)

    def export_date2csv(dataframe):
        dataframe.to_csv(PATH + filename, index=False)

    year, month, day = separate_date()
    df = convert_date2dataframe(year, month, day)
    export_date2csv(df)
