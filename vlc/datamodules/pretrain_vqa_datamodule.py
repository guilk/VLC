from vlc.datasets import VQACaptionDataset
from .datamodule_base import BaseDataModule


class PretrainVQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQACaptionDataset


    @property
    def dataset_name(self):
        return "pretrainVQA"
