import logging
import math
import os
import glob

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from .randaugment import RandAugmentMC
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.25, 0.25, 0.25)

def get_fly(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
    #    transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    #    transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    labeled_data_path = root + "/labeled/images"
    labeled_target_path = root + "/labeled/labels"
    unlabeled_data_path = root + "/unlabeled/images"
    test_data_path = root + "/test/images"
    test_target_path = root + "/test/labels"

    train_labeled_dataset = FLYSSL(
        root, labeled_data_path, labeled_target_path, labeled=True,
        transform=transform_labeled)

    train_unlabeled_dataset = FLYSSL(
        root, unlabeled_data_path,
        transform=TransformFixMatch(mean=normal_mean, std=normal_std))

    test_dataset = FLYSSL(
        root, test_data_path, test_target_path, labeled=True, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_fly_fine(args,root):

    transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
        #    transforms.Normalize(mean=normal_mean, std=normal_std)
        ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
    #    transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    labeled_data_path = root + "/labeled/images"
    labeled_target_path = root + "/labeled/labels"
    unlabeled_data_path = root + "/unlabeled/images"
    unlabeled_target_path = root + "/unlabeled/labels"
    test_data_path = root + "/labeled_test/images"
    test_target_path = root + "/labeled_test/labels"

    train_labeled_dataset = FLYSSLFINE(
        root, labeled_data_path, labeled_target_path, labeled=True,
        transform=transform_labeled)

    train_unlabeled_dataset = FLYSSLFINE(
        root, unlabeled_data_path, unlabeled_target_path,
        transform=TransformFixMatch(mean=normal_mean, std=normal_std))

    test_dataset = FLYSSLFINE(
        root, test_data_path, test_target_path, labeled=True, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

class FLYSSLFINE(Dataset):
    LABEL_MAPPING = {0: 0, 2: 1, 3: 2}  # Hard-coded remap
    excluded_labels = [1, 4]  # Labels to skip

    def __init__(self, root, img_dir, label_dir=None, labeled=False,
                 transform=None):
        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.labeled = labeled
        self.transform = transform
        
        self.data = []
        self.targets = []

        for f in os.listdir(img_dir):
            if not f.endswith('.jpg'):
                continue
            
            if self.labeled:
                label_path = os.path.join(label_dir, f.replace('.jpg', '.txt'))
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as lf:
                    line = lf.readline().strip()
                    if not line:
                        continue
                    fine_label = int(line.split()[0])
                    coarse_label = int(line.split()[1])
                    if coarse_label not in self.excluded_labels:
                        img_path = os.path.join(self.img_dir, f)
                        img = Image.open(img_path).convert('RGB')
                        self.data.append(np.array(img))
                        coarse_label = self.LABEL_MAPPING[coarse_label]
                        self.targets.append((coarse_label,fine_label))
                    else:
                        continue
            else:

                label_path = os.path.join(label_dir, f.replace('.jpg', '.txt'))
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as lf:
                    line = lf.readline().strip()
                    if not line:
                        continue
                    coarse_label = int(line.split()[0])
                    if coarse_label not in self.excluded_labels:
                        img_path = os.path.join(self.img_dir, f)
                        img = Image.open(img_path).convert('RGB')
                        self.data.append(np.array(img))
                        coarse_label = self.LABEL_MAPPING[coarse_label]
                        self.targets.append((coarse_label,-1)) #(coarse, fine)
                    else: 
                        continue

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

class FLYSSL(Dataset):
    LABEL_MAPPING = {0: 0, 2: 1, 3: 2}  # Hard-coded remap
    excluded_labels = [1, 4]  # Labels to skip

    def __init__(self, root, img_dir, label_dir=None, labeled=False,
                 transform=None):
        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.labeled = labeled
        self.transform = transform
        
        self.data = []
        self.targets = []
        for f in os.listdir(img_dir):
            if not f.endswith('.jpg'):
                continue
            
            if self.labeled:
                label_path = os.path.join(label_dir, f.replace('.jpg', '.txt'))
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as lf:
                    line = lf.readline().strip()
                    if not line:
                        continue
                    label = int(line.split()[0])
                    if label not in self.excluded_labels:
                        img_path = os.path.join(self.img_dir, f)
                        img = Image.open(img_path).convert('RGB')
                        self.data.append(np.array(img))
                        label = self.LABEL_MAPPING[label]
                        self.targets.append(label)
                    else:
                        continue
            else:
                img_path = os.path.join(self.img_dir, f)
                img = Image.open(img_path).convert('RGB')
                self.data.append(np.array(img))
                self.targets.append(-1)  # Dummy label for unlabeled data

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
        

class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=mean, std=std)
            ])
        self.transform_labeled = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                padding=int(32*0.125),
                                padding_mode='reflect'),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong), self.transform_labeled(x)
    
