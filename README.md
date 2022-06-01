# Training Vision-language Transformers from Captions Alone

This is a PyTorch/GPU implementation of the paper [VLC](https://arxiv.org/pdf/2205.09256.pdf). 
Our work is built on [MAE](https://arxiv.org/pdf/2111.06377.pdf) and the pioneering work [ViLT](https://arxiv.org/pdf/2102.03334.pdf).


## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Dataset Preparation

We follow [ViLT](https://github.com/dandelin/ViLT) and use `pyarrow` to serialize the datasets. See [this link](https://github.com/dandelin/ViLT/blob/master/DATA.md) for details.

## Pre-training

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_mlm_itm_mae per_gpu_batchsize=<BS_FITS_YOUR_GPU> whole_word_masking=True step25k image_size=384 pretrain_path=<PRETRAIN_PATH> log_dir=<LOG_FOLDER> mae_weight=1.0
```

## Fine-tuning on Downstream Tasks

### VQAv2

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_vqa_mae_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> log_dir=<LOG_FOLDER> image_size=576 learning_rate=5e-4
```

### NLVR

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_nlvr2_mae_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> log_dir=<LOG_FOLDER> image_size=384 learning_rate=5e-4
```

### COCO IR/TR

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_coco_mae_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> log_dir=<LOG_FOLDER> image_size=384 learning_rate=5e-4
```

### Flickr30K IR/TR

```bash
python run.py with data_root=<ARROW_ROOT> num_gpus=<NUM_GPUS> num_nodes=<NUM_NODES> task_finetune_irtr_f30k_mae_randaug per_gpu_batchsize=<BS_FITS_YOUR_GPU> load_path=<PRETRAINED_MODEL> log_dir=<LOG_FOLDER> image_size=384 learning_rate=5e-4
```

### Image classification on ImageNet-1K
```bash
python -m launch --nnodes=2 --nproc_per_node=16 --master_port 44875 main_finetune.py
      --batch_size 32
      --model vit_base_patch16
      --finetune <PRETRAINED_MODEL>
      --epochs 100
      --input_size 384
      --blr 5e-4
      --layer_decay 0.65
      --weight_decay 0.05
      --drop_path 0.1
      --reprob 0.25
      --mixup 0.8
      --cutmix 1.0
      --dist_eval
      --data_path <ImageNet-1K ROOT>
      --output_dir <DIR to SAVE CHECKPOINTS>
```
## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) licensed under [Apache 2.0](https://github.com/dandelin/ViLT/blob/master/LICENSE)
and [MAE](https://arxiv.org/pdf/2111.06377.pdf) under the [CC-BY-NC 4.0 license](https://github.com/facebookresearch/mae/blob/main/LICENSE).

