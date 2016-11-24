from dmc2016.data_utils import *

# class TestDataUtils(TestCase):
#     def setUp(self):
#         data_loader = DataLoader()
#         self.df = data_loader.load_test_data()
#
#     def tearDown(self):
#         pass

if __name__ == '__main__':
    data_loader = DataLoader()
    df = data_loader.load_test_data()
    # data_t = DataTransformer(df)
    # data_t.start_date_conversion()


