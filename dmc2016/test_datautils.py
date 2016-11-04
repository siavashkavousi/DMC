from unittest import TestCase

from dmc2016.datautils import *


class TestDataUtils(TestCase):
    def setUp(self):
        self.mock_df = pd.DataFrame(
            data={
                'price': [1, 1, 1, 1, 1, 1, 1, 50, 100, 200, 200, 550, 200,
                          50, 1, 500, 1, 200, 200, 200, 1, 1, 200, 1, 1, 1020, 560],
                'rrp': [5, 5, 5, 5, 5, 5, 5, 5, 20, 100, 530, 650, 60, 60,
                        60, 60, 60, 60, 60, 60, 9, 5, 3, 60, 60, 5, 20],
            }
        )

    def test_cleanup_price(self):
        df = cleanup_price(self.mock_df, lambda item_size: item_size > 5)
        self.assertEqual(len(df), 20)

    def test_cleanup_rrp(self):
        df = cleanup_rrp(self.mock_df, lambda item_size: item_size > 5)
        self.assertEqual(len(df), 20)

    def tearDown(self):
        pass
