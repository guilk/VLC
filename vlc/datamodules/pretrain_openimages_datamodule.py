from vlc.datasets import OICaptionDataset
from .datamodule_base import BaseDataModule


class OICaptionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return OICaptionDataset

    @property
    def dataset_name(self):
        return "pretrainOI"
