from sacred import Experiment

ex = Experiment("VLC")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mae": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "vlc"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # Image setting
    train_transform_keys = ["mae"]
    val_transform_keys = ["mae"]
    image_size = 384
    max_image_len = -1
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "mae_vit_base_patch16"
    hidden_size = 768
    num_heads = 12
    num_layers = 12
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False

    # mae transformer settings
    pretrain_path=""
    mask_ratio = 0.75
    use_mae = False
    mae_weight = 1.0

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 1
    precision = 16
    is_pretrain = True

# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm_mae():
    exp_name = "mlm_itm_mae"
    datasets = ["coco", "vg", "sbu", "gcc"] # The same training data as ViLT
    # datasets = ["coco", "vg", "sbu", "gcc", "pretrainVQA", "pretrainOI", "pretrainGQA", "pretrainFlickr30k", "pretrainVGQA"] # Similar training data as VinVL
    train_transform_keys = ["mae"]
    val_transform_keys = ["mae"]
    image_size = 224
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mae": 1})
    batch_size = 4096
    max_epoch = 10
    mask_ratio = 0.6
    max_image_len = 200
    use_mae = True
    mae_weight = 1.0

@ex.named_config
def task_finetune_nlvr2_mae_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["mae_randaug"]
    val_transform_keys = ["mae_test"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    is_pretrain = False

@ex.named_config
def task_finetune_vqa_mae_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["mae_randaug"]
    val_transform_keys = ["mae_test"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10
    is_pretrain = False


@ex.named_config
def task_finetune_irtr_coco_mae_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["mae_randaug"]
    val_transform_keys = ["mae_test"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = False
    draw_false_text = 15
    learning_rate = 1e-4
    is_pretrain = False


@ex.named_config
def task_finetune_irtr_f30k_mae_randaug():
    exp_name = "finetune_irtr_f30k_mae_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["mae_randaug"]
    val_transform_keys = ["mae_test"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4
    is_pretrain = False


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end

@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000

@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 100
    max_steps = 200000
    # max_steps = None
