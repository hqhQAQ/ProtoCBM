import os
import pickle
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from util.local_parts import id_to_path, name_to_id, id_to_label, id_to_attributes, cls_to_attributes, id_to_cls

class Cub2011Eval(Dataset):
    base_folder = 'test_cropped'

    def __init__(self, root, train=True, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)
        img_id = sample.img_id

        if self.transform is not None:
            img = self.transform(img)

        return img, target, img_id


class Cub2011Attribute(Dataset):
    def __init__(self, train=True, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = default_loader

        self.data_root = 'datasets/cub200_cropped_part/'
        root_prefix = 'train_cropped_augmented' if train == True else 'test_cropped'
        self.img_root = os.path.join(self.data_root, root_prefix)
        self.all_img_paths, self.all_labels, self.all_attributes = [], [], []

        all_class_dirs = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]    # Read the directories of augmented images
        for class_dir in all_class_dirs:
            img_names = os.listdir(class_dir)
            for img_name in img_names:
                img_path = os.path.join(class_dir, img_name)
                if train == True:
                    root_img_name = img_name.split('.jpg')[0].split('original_')[-1] + '.jpg'
                else:
                    root_img_name = img_name
                root_img_id = name_to_id[root_img_name]
                label = id_to_label[root_img_id]    # Targets start at 0 by default, so do not require shift
                attributes = id_to_attributes[root_img_id]

                self.all_img_paths.append(img_path)
                self.all_labels.append(label)
                self.all_attributes.append(attributes)

        self.nb_classes = 200

    def __len__(self):
        return len(self.all_img_paths)
    
    def __getitem__(self, idx):
        img_path = self.all_img_paths[idx]
        img = self.loader(img_path)
        target = self.all_labels[idx]
        attributes = self.all_attributes[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, attributes
    

class Cub2011AttributeWhole(Dataset):
    def __init__(self, train=True, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = default_loader

        self.data_root = 'datasets/cub200_cropped/'
        root_prefix = 'train_cropped_augmented' if train == True else 'test_cropped'
        self.img_root = os.path.join(self.data_root, root_prefix)
        self.all_img_paths, self.all_labels = [], []

        all_class_dirs = [os.path.join(self.img_root, x) for x in os.listdir(self.img_root)]    # Read the directories of augmented images
        for class_dir in all_class_dirs:
            img_names = os.listdir(class_dir)
            for img_name in img_names:
                img_path = os.path.join(class_dir, img_name)
                if train == True:
                    root_img_name = img_name.split('.jpg')[0].split('original_')[-1] + '.jpg'
                else:
                    root_img_name = img_name
                root_img_id = name_to_id[root_img_name]
                label = id_to_cls[root_img_id]    # Targets start at 0 by default, so do not require shift

                self.all_img_paths.append(img_path)
                self.all_labels.append(label)

        self.nb_classes = 200
        self.cls_attributes = []
        for cls_id in range(self.nb_classes):
            self.cls_attributes.append(cls_to_attributes[cls_id])

    def __len__(self):
        return len(self.all_img_paths)
    
    def __getitem__(self, idx):
        img_path = self.all_img_paths[idx]
        img = self.loader(img_path)
        target = self.all_labels[idx]
        # attributes = self.all_attributes[idx]
        attributes = self.cls_attributes[target]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, attributes