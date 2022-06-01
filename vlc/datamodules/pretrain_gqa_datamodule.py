from vlc.datasets import GQACaptionDataset
from .datamodule_base import BaseDataModule


class PretrainGQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return GQACaptionDataset


    @property
    def dataset_name(self):
        return "pretrainGQA"
