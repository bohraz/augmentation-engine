import torch
import os
from torch.utils.data import Dataset
from filament_augmentation.generator._filament_generator import _FilamentGenerator
from filament_augmentation.metadata.filament_metadata import FilamentMetadata


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
        class_id = torch.tensor([class_name])
        return anno_tensor, class_id


if __name__ == "__main__":
    bbso_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..', 'petdata', 'bbso_json_data', '2015_chir_data.json'))
    transforms_json = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..', 'petdata', 'input_transformations', 'transforms.json'))
    bbso_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'petdata', '2015'))
    dataset = FilamentDataset(bbso_path,bbso_json,"2015-08-05 17:36:15", "2015-08-11 18:15:17")
    print(type(dataset.data))
    # data_loader = FilamentDataLoader(dataset, 6, (1, 1, 1), 10)
    # print(len(data_loader))
    # for imgs, labels in data_loader:
    #     print("Batch of images has shape: ", imgs)
    #     print("Batch of labels has shape: ", labels)