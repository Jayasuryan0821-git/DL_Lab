import glob
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, transform, dataset_type="train"):
        self.imgs_path = f"./data/cats_and_dogs_filtered/{dataset_type}/"
        file_list = glob.glob(self.imgs_path + "*")
        print("File list", file_list)
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for img_path in glob.glob(class_path + "/*.jpg"):
                self.data.append([img_path, class_name])
        print(self.data)
        self.class_map = {"dogs": 0, "cats": 1}
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
