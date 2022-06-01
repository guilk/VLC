from vlc.datasets import Flickr30kCaptionDataset
from .datamodule_base import BaseDataModule


class PretrainF30kDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Flickr30kCaptionDataset


    @property
    def dataset_name(self):
        return "pretrainFlickr30k"
