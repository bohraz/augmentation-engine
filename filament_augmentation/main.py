import os
import time
import matplotlib.pyplot as plt
from torchvision import transforms

from filament_augmentation.loader.filament_dataloader import FilamentDataLoader
from filament_augmentation.generator.filament_dataset import FilamentDataset
from filament_augmentation.metadata.filament_metadata import FilamentMetadata
from filament_augmentation.transforms.heliographic_transformation import Heliographic

__file__ = 'Demo.ipynb'
path_to_json =  os.path.abspath(os.path.join(r'C:\Users\jaibr\OneDrive - University of Missouri\Resea\environment','2015_chir_data2.json'))
# path to my 2015 folder is at /home/user/data/2015 , therefore path_to_your_2015_folder = '/home/user/data/'
path_to_2015_folder = bbso_path = os.path.abspath(r"C:\Users\jaibr\OneDrive - University of Missouri\Resea\environment")
print(path_to_json)
print(path_to_2015_folder)

# Use MIRO or LucidChart for class chart

filamentInfo = FilamentMetadata(ann_file = path_to_json, start_time = '2015-08-01 00:00:15',
                                    end_time = '2015-08-30 23:59:59')
filamentInfo.get_chirality_distribution()

dataset = FilamentDataset(bbso_path = path_to_2015_folder, ann_file = path_to_json,
                          start_time = "2015-08-01 17:36:15", end_time = "2015-08-09 17:36:15")
#whatever goes inside of tr, so somewhere in the code you should see that heliographic is a transformansforms1 is a transform,
#annotations = filamentInfo.bbso_json['annotations']

transforms1 = [
    Heliographic(15, 15),
    transforms.ColorJitter(brightness=(0.25,1.25), contrast=(0.25,2.00), saturation=(0.25,2.25)),
    transforms.RandomRotation(15,expand=False,fill=110)
]

data_loader = FilamentDataLoader(dataset = dataset,batch_size = 1 , filament_ratio = (0, 0, 1),
                                 n_batchs = 10, transforms = transforms1, image_dim = -1)

for original_imgs, transformed_imgs, labels in data_loader:
    for org_img, img, label in zip(original_imgs ,transformed_imgs, labels):
        print("Original image")
        plt.imshow(org_img, cmap='gray')
        plt.show()
        print("Transformed image")
        plt.imshow(img, cmap='gray')
        plt.show()
        print("Label",label)
    break


# C:\Users\jaibr\OneDrive - University of Missouri\Resea\environment\2015_chir_data2.json
# C:\Users\jaibr\OneDrive - University of Missouri\Resea\environment
# loading annotations into memory...
# Done (t=0.05s)
# creating index...
# index created!
# Traceback (most recent call last):
#   File "c:\Users\jaibr\OneDrive - University of Missouri\Resea\environment\test.py", line 36, in <module>
#     for original_imgs, transformed_imgs, labels in data_loader:
#   File "C:\Python312\Lib\site-packages\torch\utils\data\dataloader.py", line 630, in __next__
#     data = self._next_data()
#            ^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\loader\filament_dataloader.py", line 112, in _next_data
#     data = self._dataset_fetcher.fetch(index)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\loader\filament_dataloader.py", line 143, in fetch
#     return self.collate_fn(data)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\loader\filament_dataloader.py", line 54, in collate_fn
#     c, rm = filament_augmentation.save_filaments()
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\augment\augmentation.py", line 46, in save_filaments
#     self.__generate_filaments_each_type(_LEFT_CHIRALITY, n_times * self.n_l)
#   File "C:\Python312\Lib\site-packages\filament_augmentation\augment\augmentation.py", line 118, in __generate_filaments_each_type
#     count, idx = self.__image_augments_original(chirality_type, count, idx, n, n_image_copies)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\augment\augmentation.py", line 157, in __image_augments_original
#     count = self.__generate_image(c, chirality_type, count, id)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\augment\augmentation.py", line 200, in __generate_image
#     self.augmented_data.append(self.__perform_transformations
#                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\augment\augmentation.py", line 216, in __perform_transformations
#     transformed_image = transform.transform_image()
#                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\transforms\transformation.py", line 25, in transform_image
#     transformed_img = self._transforms(self.image)
#                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Python312\Lib\site-packages\torchvision\transforms\transforms.py", line 95, in __call__
#     img = t(img)
#           ^^^^^^
#   File "C:\Python312\Lib\site-packages\filament_augmentation\transforms\heliographic_transformation.py", line 44, in __call__
#     img, ann_obj, bbox = data
#     ^^^^^^^^^^^^^^^^^^
# TypeError: cannot unpack non-iterable Image object

#Working Transformations
# FilamentDataLoader Initialization transforms [ColorJitter(brightness=(0.25, 1.25), contrast=(0.25, 2.0), saturation=(0.25, 2.25), hue=None), RandomRotation(degrees=[-15.0, 15.0], interpolation=nearest, expand=False, fill=110)]
# Transform List: [ColorJitter(brightness=(0.25, 1.25), contrast=(0.25, 2.0), saturation=(0.25, 2.25), hue=None), RandomRotation(degrees=[-15.0, 15.0], interpolation=nearest, expand=False, fill=110)]
# Transform List converted to _Transformation Object: <filament_augmentation.transforms.transformation._Transformation object at 0x00000229AB0964B0>
# Image to transform: <PIL.Image.Image image mode=L size=134x44 at 0x229AB096480>
# Transform Object from Before: Compose(
#     ColorJitter(brightness=(0.25, 1.25), contrast=(0.25, 2.0), saturation=(0.25, 2.25), hue=None)
#     RandomRotation(degrees=[-15.0, 15.0], interpolation=nearest, expand=False, fill=110)
# )
# Transformed Image: <PIL.Image.Image image mode=L size=134x44 at 0x229AB096420>

#Heliographic Transformation
# FilamentDataLoader Initialization transforms [<filament_augmentation.transforms.heliographic_transformation.Heliographic object at 0x0000027388854080>]
# Transform List: [<filament_augmentation.transforms.heliographic_transformation.Heliographic object at 0x0000027388854080>]
# Transform List converted to _Transformation Object: <filament_augmentation.transforms.transformation._Transformation object at 0x0000027388857050>
# Image to transform: <PIL.Image.Image image mode=L size=76x67 at 0x27385F8CF20>
# Transform Object from Before: Compose(
#     <filament_augmentation.transforms.heliographic_transformation.Heliographic object at 0x0000027388854080>
# )
# Heliographic call data(list expected) <PIL.Image.Image image mode=L size=76x67 at 0x27385F8CF20>

# Note missing Transformed Image print statement, that is because of the error.