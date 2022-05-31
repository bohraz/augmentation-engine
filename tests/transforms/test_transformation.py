import unittest
import os

import PIL
from filament_augmentation.utils.image_utils import get_image
from filament_augmentation.transforms.transformation import _Transformation
from filament_augmentation.transforms import transformation
from torchvision import transforms

class TestTransforms(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.transforms_list = [
            transforms.RandomRotation(15, expand=False, fill=110)
        ]

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_get_transforms(self):
        """Test if list of transforms objects is convert to transforms.compose list"""
        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","test_data","transforms","tranform_test_image.jpg"))
        image = get_image(image_path)
        transformation = _Transformation(image, transformation.get_transform(self.transforms_list))
        transformed_img = transformation.transform_image()
        self.assertIsInstance(transformed_img, PIL.Image.Image)

    def test_get_transforms_t1(self):
        """Test if list of non-transforms objects throws error"""
        transform_compose = transformation.get_transform(self.transforms_list)
        self.assertIsInstance(transform_compose, transforms.Compose)
