from torch.utils.data import DataLoader

from data.xcad import DatasetXCAD
from data.dca1 import DatasetDCA1
from data.chuac import DatasetCHUAC

class CSDataset:

    @classmethod
    def initialize(cls, datapath):

        cls.datasets = {
            'xcad': DatasetXCAD,
            'dca1': DatasetDCA1,
            'chuac': DatasetCHUAC
        }

        cls.datapath = datapath

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, img_mode, img_size):
        shuffle = split == 'train'
        nworker = nworker 

        if split == 'train':
            dataset = cls.datasets[benchmark](benchmark, 
                                              datapath=cls.datapath,
                                              split=split, 
                                              img_mode=img_mode,
                                              img_size=img_size)
        else:
            dataset = cls.datasets[benchmark](benchmark,
                                              datapath=cls.datapath,
                                              split=split, 
                                              img_mode='same',
                                              img_size=img_size) 

        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)

        return dataloader
