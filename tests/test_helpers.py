import unittest
import DLBio.helpers as helpers
import pandas as pd


class TestSubtable(unittest.TestCase):
    def test_subtable(self):
        df = pd.DataFrame({
            'x': ['A', 'B', 'A', 'C', 'A'],
            'y': [3, 4, 5, 4, 4]
        })

        subtable = helpers.get_sub_dataframe(df, {'x': 'A', 'y': [3, 4]})

        self.assertTrue(set(subtable['x']) == {'A'})
        self.assertTrue(set(subtable['y']) == {3, 4})


if __name__ == '__main__':
    unittest.main()
