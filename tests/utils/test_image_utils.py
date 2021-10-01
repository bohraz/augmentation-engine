import unittest
import os

import PIL

from filament_augmentation.utils import image_utils

class TestImageUtilies(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","test_data","transforms","tranform_test_image.jpg"))

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_image_t1(self):
        """Send path and get image"""
        image = image_utils.get_image(self.image_path)
        self.assertIsInstance(image, PIL.Image.Image)



