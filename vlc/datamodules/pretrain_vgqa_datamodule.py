from vlc.datasets import VGQACaptionDataset
from .datamodule_base import BaseDataModule


class VGQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VGQACaptionDataset

    @property
    def dataset_name(self):
        return "pretrainVGQA"
