import numpy
import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class city_scapes(Dataset):
    def __init__(self, datapath, transform):
        self.dir_path = datapath
        self.image_path = self.dir_path
        print(self.image_path)
        self.filtered_images, self.filtered_filenames = self.get_filtered_data()
        self.labels = self.get_labels(self.filtered_filenames)
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.samples = {'image': self.filtered_images[idx], 'label': self.labels[idx]}
        if self.transform:
            self.samples['image'] = self.transform(self.samples['image'])
        return self.samples

    def __len__(self):
        return len(self.filtered_images)


    def image_dims(self, image):
        w, h = image.size
        if w * h >= 900:
            return True
        else:
            return False

    def get_labels(self, filename):
        labels = []
        _MAP_CS_TO_TR_LABEL = {24: 0, 25: 1, 26: 2}
        for name in filename:
            id_ = int(name[-9:-4])  # Get last part of filename which reveals label of image
            base_id = id_ if (id_ < 1000) else id_ // 1000
            labels.append(_MAP_CS_TO_TR_LABEL[base_id])
        return labels

    def get_filtered_data(self):
        images = []
        file_names = []
        label_list = []
        print("fetching data from the data directory")

        for filename in os.listdir(self.image_path):
            img = Image.open(os.path.join(self.image_path, filename)).convert('RGB')
            if img is not None:
                if self.image_dims(img):
                    images.append(img)
                    file_names.append(filename)
        print('Number of Images', len(images))
        print('Number of file names', len(file_names))
        return images, file_names
