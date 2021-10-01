import unittest
import os
from filament_augmentation.metadata.filament_metadata import FilamentMetadata
from datetime import datetime

class TestMetadata(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.path_to_ann_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'test_data', 'ann.json'))

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_chirality_distribution_exact_timestamps(self):
        """Test when timestamps are exactly same as in annotation file"""
        metadata = FilamentMetadata(self.path_to_ann_file,"2015-08-01 17:48:14", "2015-08-31 18:18:39")
        distribution = metadata.get_chirality_distribution()
        self.assertEqual(distribution,(0, 2, 2))

    def test_distribution_non_exact_starttime(self):
        """Test when start timestamp is not exact as annotation file"""
        metadata = FilamentMetadata(self.path_to_ann_file, "2015-08-20 17:50:14", "2015-08-31 18:18:39")
        distribution = metadata.get_chirality_distribution()
        self.assertEqual(distribution, (0, 0, 1))

    def test_distribution_non_exact_endtime(self):
        """Test when end timestamp is not exact as annotation file"""
        metadata = FilamentMetadata(self.path_to_ann_file, "2015-08-01 17:48:14", "2015-08-01 18:18:39")
        distribution = metadata.get_chirality_distribution()
        self.assertEqual(distribution, (0, 2, 1))