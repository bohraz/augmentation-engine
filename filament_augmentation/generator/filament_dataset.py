import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms

from filament_augmentation.generator._filament_generator import _FilamentGenerator
from filament_augmentation.metadata.filament_metadata import FilamentMetadata
from filament_augmentation.loader.filament_dataloader import FilamentDataLoader

class FilamentDataset(Dataset):

    def __init__(self, bbso_path: str, ann_file: str, start_time: str, end_time: str):
        """
        The constructor gets the image ids based on start and end time.
        based on the image ids, filaments annotation index and their respective class labels
        are initialized to dataset.
        :param bbso_path: path to bsso full disk images.
        :param ann_file: path to annotations file.
        :param start_time: start time in YYYY:MM:DD HH:MM:SS.
        :param end_time: end time in YYYY:MM:DD HH:MM:SS.
        """
        filament_metadata = FilamentMetadata(ann_file,start_time, end_time)
        filament_metadata.parse_data()
        self.bbso_img_ids: list = filament_metadata.bbso_img_ids
        self.filament_cutouts_data: _FilamentGenerator = _FilamentGenerator(ann_file, bbso_path, self.bbso_img_ids)
        self.data: list = self.filament_cutouts_data.filament_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anno, class_name = self.data[idx]
        anno_tensor = torch.from_numpy(anno)
        class_id = torch.tensor(class_name)
        return anno_tensor, class_id


if __name__ == "__main__":
    transforms1 = [
        transforms.ColorJitter(brightness=(0.25, 1.25), contrast=(0.25, 2.00), saturation=(0.25, 2.25)),
        transforms.RandomRotation(15, expand=False, fill=110)
    ]
    bbso_json = r'D:\GSU_Assignments\Summer_sem\filaments_data_augmentation\chir_data.json'
    transforms_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..', 'petdata', 'input_transformations', 'transforms.json'))
    bbso_path = r'D:\GSU_Assignments\Summer_sem\RA\bbso_data_retriever\bbso_fulldisk'
    dataset = FilamentDataset(bbso_path,bbso_json,start_time = "2000-01-01 00:00:00", end_time = "2014-12-31 11:59:59")
    print(type(dataset.data))
    data_loader = FilamentDataLoader(dataset,9, (1, 1, 1), 5, transforms=transforms1, image_dim= 256)
    # print(len(data_loader))
    for _,imgs, labels in data_loader:
        print("Batch of images has shape: ", imgs.shape)
        print("Batch of labels has shape: ", labels)