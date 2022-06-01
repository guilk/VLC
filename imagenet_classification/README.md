## Finetune VLC model on ImageNet-1K dataset

To install python packages, follow **steup** section in **mfinetune_full_imagenet.yaml**.

Before finetuning on ImageNet-1K dataset, you need to convert the keys of VLC model to MAE-style:
```
python convert_vlc2mae.py --vlc_model PATH/TO/VLC/MODEL --mae_model FOLDER/PATH/TO/MAE/MODEL
```

To finetune ViT-B/16 with **multi-node distributed training**, run the following on 1 nodes with 16 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=16 main_finetune.py
      --batch_size 64
      --model vit_base_patch16
      --finetune PATH/TO/MAE/MODEL
      --epochs 100
      --blr 5e-4
      --layer_decay 0.65
      --weight_decay 0.05
      --drop_path 0.1
      --reprob 0.25
      --mixup 0.8
      --cutmix 1.0
      --dist_eval
      --data_path DATA/ROOT/TO/ImageNet-1K
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 1 (`nodes`) * 16 (gpus per node) = 1024. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- We have run two trials with different random seeds. The results are 84.5 and 84.4.

To evaluate ViT-B/16 with **multi-node inference**, run the following on 1 nodes with 16 GPUs each:
```
python -m torch.distributed.launch --nproc_per_node=16 main_finetune.py
      --batch_size 64
      --model vit_base_patch16
      --finetune PATH/TO/MAE/MODEL
      --eval
      --input_size 384
      --data_path DATA/ROOT/TO/ImageNet-1K
```
