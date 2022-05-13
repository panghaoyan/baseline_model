import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import xml.etree.ElementTree as ElementTree
from argparse import ArgumentParser

# Labels = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK", "VA", "ND"]
Labels = ["Crack breaks and collapes","Surface damage","Production error","Deformation","Displaced joint","Intruding sealing material",
          "Roots","Infiltration","Settled deposits","Attached deposits","Obstacle","Branch pipe","Chiseled Connection","Drilled Connection",
          "Lateral reinstatement","Connection with transition profile","Connection with construction changes", "VA", "Normal pipe"]


class MultiLabelDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        # self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)
        self.loadAnnotations()
        self.class_weights = self.getClassWeights()
        self.labels_batch = []

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        if self.split == 'Test':
            gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=["Filename"], engine='python')
        else:
            gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename", "Defect"], engine='python')

        if self.onlyDefects:
            gt = gt[gt["Defect"] == 1]

        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        self.labels_fullname = gt[self.LabelNames].columns
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]
        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index, :]

        label_current_name = []

        for index_list, value in enumerate(target):
             if value == 1:
                 label_current_name.append(self.labels_fullname[index_list])
             else:
                 label_current_name.append('')
        self.labels_batch.append(label_current_name)


        return img, target

    def getClassWeights(self):
        data_len = self.labels.shape[0]
        class_weights = []

        for defect in range(self.num_classes):
            pos_count = len(self.labels[self.labels[:,defect] == 1])
            neg_count = data_len - pos_count

            class_weight = neg_count/pos_count if pos_count > 0 else 0
            class_weights.append(np.asarray([class_weight]))
        return torch.as_tensor(class_weights).squeeze()


class MultiLabelDatasetInference(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, onlyDefects=False):
        super(MultiLabelDatasetInference, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.onlyDefects = onlyDefects

        self.num_classes = len(self.LabelNames)

        self.loadAnnotations()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename"])

        self.imgPaths = gt["Filename"].values
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        return img, path


class BinaryRelevanceDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader, defect=None):
        super(BinaryRelevanceDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.LabelNames = Labels.copy()
        self.LabelNames.remove("VA")
        self.LabelNames.remove("ND")
        self.defect = defect

        assert self.defect in self.LabelNames

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", self.defect])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt[self.defect].values.reshape(self.imgPaths.shape[0], 1)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)


class BinaryDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Train", transform=None, loader=default_loader):
        super(BinaryDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = default_loader

        self.num_classes = 1

        self.loadAnnotations()
        self.class_weights = self.getClassWeights()

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename", "Defect"])

        self.imgPaths = gt["Filename"].values
        self.labels =  gt["Defect"].values.reshape(self.imgPaths.shape[0], 1)
        print(self.labels.shape)
        
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = self.labels[index]

        return img, target, path

    def getClassWeights(self):
        pos_count = len(self.labels[self.labels == 1])
        neg_count = self.labels.shape[0] - pos_count
        class_weight = np.asarray([neg_count/pos_count])

        return torch.as_tensor(class_weight)


class CODEBRIMSplit(datasets.ImageFolder):
    def __init__(self, root, xml_list, transform=None, target_transform=None, loader=datasets.folder.default_loader):
        super(CODEBRIMSplit, self).__init__(root, transform, target_transform, loader)
        self.file_list = {}
        self.num_classes = 6
        for file_name in xml_list:
            last_dot_idx = file_name.rfind('.')
            f_name_idx = file_name.rfind('/')
            root_path = file_name[f_name_idx + 1: last_dot_idx]
            tree = ElementTree.parse(file_name)
            root = tree.getroot()
            for defect in root:
                crop_name = list(defect.attrib.values())[0]
                target = self.compute_target_multi_target(defect)
                self.file_list[os.path.join(root_path, crop_name)] = target

    def __getitem__(self, idx):
        image_batch = super(CODEBRIMSplit, self).__getitem__(idx)[0]
        image_name = self.imgs[idx][0]
        f_name_idx = image_name.rfind('/')
        f_dir_idx = image_name[: f_name_idx].rfind('/')
        de_lim = image_name.rfind('_-_')
        file_type = image_name.rfind('.')
        if de_lim != -1:
            name = image_name[f_dir_idx + 1: de_lim] + image_name[file_type:]
        else:
            name = image_name[f_dir_idx + 1:]
        return [image_batch, self.file_list[name]]

    def compute_target_multi_target(self, defect):
        out = np.zeros(self.num_classes, dtype=np.float32)
        for i in range(self.num_classes):
            if defect[i].text == '1':
                out[i] = 1.0
        return out


class CODEBRIM:
    def __init__(self, is_gpu, path, img_size=224, batch_size=32, workers=4):
        self.num_classes = 6
        self.dataset_path = path
        self.batch_size = batch_size
        self.workers = workers
        self.img_size = img_size
        self.dataset_xml_list = [os.path.join(self.dataset_path, 'metadata/background.xml'),
                                 os.path.join(self.dataset_path, 'metadata/defects.xml')]
        self.train_set, self.val_set, self.test_set = self.get_dataset(self.img_size)
        self.train_loader, self.val_loader, self.test_loader = self.get_dataset_loader(self.batch_size, self.workers,
                                                                                       is_gpu)

    def get_dataset(self, img_size):
        train_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'train'),
                                  self.dataset_xml_list,
                                  transform=transforms.Compose([transforms.Resize((img_size, img_size)),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.523, 0.453, 0.345], std=[0.210, 0.199, 0.154])
                                                                ])
                                  )
        val_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'val'),
                                self.dataset_xml_list,
                                transform=transforms.Compose([transforms.Resize(img_size),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.523, 0.453, 0.345],std=[0.210, 0.199, 0.154])]))
        test_set = CODEBRIMSplit(os.path.join(self.dataset_path, 'test'),
                                 self.dataset_xml_list,
                                 transform=transforms.Compose([transforms.Resize(img_size),
                                                               transforms.CenterCrop(img_size),
                                                               transforms.ToTensor()]))
        return train_set, val_set, test_set

    def get_dataset_loader(self, batch_size, workers, is_gpu, transform=None):
        train_loader = torch.utils.data.DataLoader(self.train_set, num_workers=workers, batch_size=batch_size,
                                                   shuffle=True, pin_memory=is_gpu)
        val_loader = torch.utils.data.DataLoader(self.val_set, num_workers=workers, batch_size=batch_size,
                                                 shuffle=False, pin_memory=is_gpu)
        test_loader = torch.utils.data.DataLoader(self.test_set, num_workers=workers, batch_size=batch_size,
                                                  shuffle=False, pin_memory=is_gpu)

        return train_loader, val_loader, test_loader

