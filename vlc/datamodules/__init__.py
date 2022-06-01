from .vg_caption_datamodule import VisualGenomeCaptionDataModule
from .f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .conceptual_caption_datamodule import ConceptualCaptionDataModule
from .sbu_datamodule import SBUCaptionDataModule
from .vqav2_datamodule import VQAv2DataModule
from .nlvr2_datamodule import NLVR2DataModule
from .pretrain_vqa_datamodule import PretrainVQADataModule
from .pretrain_openimages_datamodule import OICaptionDataModule
from .pretrain_gqa_datamodule import PretrainGQADataModule
from .pretrain_flickr30k_datamodule import PretrainF30kDataModule
from .pretrain_vgqa_datamodule import VGQADataModule

_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "pretrainVQA": PretrainVQADataModule,
    "pretrainOI": OICaptionDataModule,
    "pretrainGQA": PretrainGQADataModule,
    "pretrainFlickr30k": PretrainF30kDataModule,
    "pretrainVGQA":VGQADataModule,
}
