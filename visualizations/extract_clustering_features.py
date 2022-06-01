import torch
import copy
import numpy as np

from PIL import Image

from vlc.config import ex
from vlc.modules import VLCTransformer

from vlc.transforms import mae_transform_test
from vlc.datamodules.datamodule_base import get_pretrained_tokenizer

import pickle

def extract_demo_features(img_path, mp_text, model, device, tokenizer):
    image = Image.open(img_path).convert("RGB")
    img = mae_transform_test(size=384)(image)

    img = img.unsqueeze(0).to(device)

    batch = {"text": [""], "image": [None]}
    inferred_token = [mp_text]
    batch["image"][0] = img

    encoded = tokenizer(inferred_token,
                        truncation=True,
                        max_length=40)

    with torch.no_grad():
        batch["text"] = inferred_token
        batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
        batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
        infer = model(batch)
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        img_features = np.squeeze(img_emb.detach().cpu().numpy())
        valid_features = img_features[1:,:]
    return valid_features

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)

    loss_names = {
        "itm": 1,
        "mlm": 1,
        "mpp": 0,
        "vqa": 0,
        "imgcls": 0,
        "nlvr2": 0,
        "irtr": 0,
        "arc": 0,
        "snli":0,
        "mae":1,
    }
    tokenizer = get_pretrained_tokenizer(_config["tokenizer"])

    _config.update(
        {
            "load_path": "/home/liangkeg/internship/mim/ViLT_mae_models/mvlm_epoch100.ckpt",
            "is_pretrain": False,
            "test_only":True,
        }
    )
    _config.update(
        {
            "loss_names": loss_names,
        }
    )

    print(_config)
    model = VLCTransformer(_config)
    model.setup("test")
    model.eval()

    device = "cuda:3" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    img_path = './livingroom.jpeg'
    caption = 'a room with a rug, a chair, a painting, and a plant.'
    valid_features = extract_demo_features(img_path, caption, model,device, tokenizer)
    with open('./ours_livingroom_feats.pkl', 'wb') as output:
        pickle.dump(valid_features, output)