# Guomin @2022/09/24
import os.path as osp
import numpy as np
import torch
from .base_dataset import BaseDataset

class BCNet_ClothDataset(BaseDataset):
    def __init__(self, options, dataset, use_augmentation=True, is_train=True):
        super().__init__(options, dataset, use_augmentation, is_train)
        self.img_ids = self.data['img_ids'].astype(np.int32)
        self.gar_part = 'up'
        if options.garment == 'skirts' or options.garment == 'short_skirts' or options.garment == 'pants':
            self.gar_part = 'bottom'

    def __getitem__(self, index):
        item = super().__getitem__(index)
        img_id = self.img_ids[index]
        labels_data = np.load(osp.join(self.img_dir, 'motion_datas', 'all_train_datas', f'{img_id}.npz'))
        item['gar_vs'] = labels_data[self.gar_part]
        item['gar_tran'] = labels_data['tran'].reshape(1,3)
        return item

    def ___len__(self):
        return super().__len__()

class BCNetDataset(torch.utils.data.Dataset):
    def __init__(self, options) -> None:
        super().__init__()
        self.ds = BCNet_ClothDataset(options, options.dataset)
        self.len = len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]

    def __len__(self):
        return self.len