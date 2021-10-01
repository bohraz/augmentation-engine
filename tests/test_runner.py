import unittest
from .utils import test_image_utils
from .utils import test_timestamp_utils
from .metadata import test_metadata
from .transforms import test_transformation


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    suite_utils = unittest.TestSuite()
    suite_metadata = unittest.TestSuite()
    suite_transforms = unittest.TestSuite()

    suite_utils.addTests(test_loader.loadTestsFromModule(test_image_utils))
    suite_utils.addTests(test_loader.loadTestsFromModule(test_timestamp_utils))
    suite_transforms.addTests(test_loader.loadTestsFromModule(test_transformation))
    suite_metadata.addTests(test_loader.loadTestsFromModule(test_metadata))

    all_test_suites = unittest.TestSuite([suite_utils, suite_transforms, suite_metadata])

    runner = unittest.TextTestRunner(verbosity=3)
    result = runner.run(all_test_suites)