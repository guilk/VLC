import torch
import copy
import time
import io
import numpy as np
import re
import pickle
import ipdb
import os

from PIL import Image

from vlc.config import ex
from vlc.modules import VLCTransformer

from vlc.modules.objectives import cost_matrix_cosine, ipot
from vlc.transforms import mae_transform_test
from vlc.datamodules.datamodule_base import get_pretrained_tokenizer


def visualize_result(img_path, mp_text, token_infos, model, device, tokenizer):
    image_name = img_path.split('/')[-1].split('.')[0]
    image = Image.open(img_path).convert("RGB")
    img = mae_transform_test(size=384)(image)
    img = img.unsqueeze(0).to(device)

    batch = {"text": [""], "image": [None]}
    tl = len(re.findall("\[MASK\]", mp_text))
    inferred_token = [mp_text]
    batch["image"][0] = img

    with torch.no_grad():
        for i in range(tl):
            batch["text"] = inferred_token
            encoded = tokenizer(inferred_token)
            batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
            batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
            encoded = encoded["input_ids"][0][1:-1]
            infer = model(batch)
            mlm_logits = model.mlm_score(infer["text_feats"])[0, 1:-1]
            mlm_values, mlm_ids = mlm_logits.softmax(dim=-1).max(dim=-1)
            mlm_values[torch.tensor(encoded) != 103] = 0
            select = mlm_values.argmax().item()
            encoded[select] = mlm_ids[select].item()
            inferred_token = [tokenizer.decode(encoded)]

    selected_token = ""
    encoded = tokenizer(inferred_token,
                        truncation=True,
                        max_length=40)

    bs,nc,H,W = 1,1,24,24
    patch_index = (
        torch.stack(
            torch.meshgrid(
                torch.arange(H), torch.arange(W)
            ),
            dim=-1,
        )[None, None, :, :, :]
            .expand(bs, nc, -1, -1, -1)
            .flatten(1, 3)
    )

    try:
        for token_info in token_infos:
            hidx, token, pos = token_info[0], token_info[1], token_info[2]
            selected_token = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][hidx])
            if pos != 'VERB' or selected_token != token:
                continue
            print(selected_token, token)

            if hidx > 0 and hidx < len(encoded["input_ids"][0][:-1]):
                with torch.no_grad():
                    batch["text"] = inferred_token
                    batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
                    batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
                    batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
                    infer = model(batch)
                    txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
                    txt_mask, img_mask = (
                        infer["text_masks"].bool(),
                        infer["image_masks"].bool(),
                    )
                    for i, _len in enumerate(txt_mask.sum(dim=1)):
                        txt_mask[i, _len - 1] = False
                    txt_mask[:, 0] = False
                    img_mask[:, 0] = False
                    txt_pad, img_pad = ~txt_mask, ~img_mask


                    cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
                    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
                    cost.masked_fill_(joint_pad, 0)

                    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
                        dtype=cost.dtype
                    )
                    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
                        dtype=cost.dtype
                    )
                    T = ipot(
                        cost.detach(),
                        txt_len,
                        txt_pad,
                        img_len,
                        img_pad,
                        joint_pad,
                        0.1,
                        1000,
                        1,
                    )

                    plan = T[0]
                    plan_single = plan * len(txt_emb)
                    cost_ = plan_single.t()

                    cost_ = cost_[hidx][1:].cpu()

                    heatmap = torch.zeros(H, W)
                    for i, pidx in enumerate(patch_index[0]):
                        h, w = pidx[0].item(), pidx[1].item()
                        heatmap[h, w] = cost_[i]

                    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
                    heatmap = np.clip(heatmap, 1.0, 3.0)
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                    _w, _h = image.size
                    overlay = Image.fromarray(np.uint8(heatmap * 255), "L").resize(
                        (_w, _h), resample=Image.NEAREST
                    )
                    image_rgba = image.copy()
                    image_rgba.putalpha(overlay)
                    image = image_rgba

            save_img = Image.fromarray(np.array(image), 'RGBA')
            dst_path = os.path.join('./ours_demo', '{}_{}_{}.png'.format(image_name, selected_token, hidx))
            save_img.save(dst_path)
            cmd = 'scp {} {}'.format(img_path, './ours_demo/{}.jpg'.format(image_name))
            os.system(cmd)
    except:
        print('Error processing image')

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

    device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
    model.to(device)

    with open('./visual_data.pkl', 'rb') as input:
        data = pickle.load(input)


    select_root = './ood'
    files = os.listdir(select_root)
    img_names = [file_name for file_name in files if file_name.endswith('.jpg')]

    for img_index, image_name in enumerate(img_names):
        print('process {}th of {} image'.format(img_index, len(img_names)))
        img_path, caption, token_infos = data[image_name]
        visualize_result(img_path, caption, token_infos, model, device, tokenizer)