import unittest
import os

from sortedcontainers import SortedDict
from datetime import datetime

from filament_augmentation.utils import timestamp_utilites
from filament_augmentation.utils import file_utilities

class TestTimestampUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.path_to_ann_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'..','test_data','ann.json'))
        cls.time_dict = SortedDict({
            2015080117481400: datetime(2015, 8, 1, 17, 48, 14),
            2015080117481401: datetime(2015, 8, 1, 17, 48, 14),
            2015080117481402: datetime(2015, 8, 1, 17, 48, 14),
            2015083118183927: datetime(2015, 8, 31, 18, 18, 39)})

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_closest_t1(self):
        """Test get closest for exact value returns the closest index"""
        index = timestamp_utilites.find_closest(datetime(2015, 8, 1, 17, 48, 14),self.time_dict)
        expected_index = 0
        self.assertEqual(index,expected_index)

    def test_get_closest_t2(self):
        """Test get closest returns the closest index"""
        index = timestamp_utilites.find_closest(datetime(2015, 8, 31, 20, 18, 39), self.time_dict)
        expected_index = 3
        self.assertEqual(index, expected_index)

    def test_get_timestamp_dict(self):
        """"""
        bbso_json = file_utilities.read_json(self.path_to_ann_file)
        actual_dict = timestamp_utilites.get_timestamp_dict(bbso_json)
        expected_dict = self.time_dict
        self.assertDictEqual(actual_dict, expected_dict)