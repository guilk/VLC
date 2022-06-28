import os
import json
import wget
import spacy
import pickle
import numpy as np

from PIL import Image
import torch
import copy
from vlc.config import ex
from vlc.modules import VLCTransformer

from vlc.modules.objectives import cost_matrix_cosine, ipot
from vlc.transforms import mae_transform_test
from vlc.datamodules.datamodule_base import get_pretrained_tokenizer


def collect_data():
    nlp = spacy.load("en_core_web_sm")
    target_pos = {'ADJ', 'NOUN', 'VERB'}
    visual_dict = {}

    file_path = '../../nocaps/nocaps_val_4500_captions.json'
    data = json.load(open(file_path, 'r'))

    images = data['images']
    annotations = data['annotations']

    id_names = {}
    for img_info in images: # keys: file_name, coco_url, open_images_id, id, domain
        url = img_info['coco_url']
        domain = img_info['domain']
        id_names[img_info['id']] = '{}/{}'.format(domain, img_info['file_name'])

    for index, ann in enumerate(annotations): # keys: image_id, id, caption
        image_id, id, caption = ann['image_id'], ann['id'], ann['caption']
        doc = nlp(caption)
        target_tokens = []
        for token_index, token in enumerate(doc):
            if token.pos_ in target_pos:
                target_tokens.append((token_index+1, token.text, token.pos_))

        print(index, ann['image_id'], ann['id'], ann['caption'], target_tokens)
        if image_id not in visual_dict:
            visual_dict[image_id] = []
        visual_dict[image_id].append((id, caption, target_tokens))

    with open('./nocaps_visual_raw.pkl', 'wb') as output:
        pickle.dump(visual_dict, output)

    with open('./nocaps_id_names.pkl', 'wb') as output:
        pickle.dump(id_names, output)


def compute_heat_maps(img_path, mp_text, token_infos, model, device, tokenizer):
    image_name = img_path.split('/')[-1].split('.')[0]
    image = Image.open(img_path).convert("RGB")
    img = mae_transform_test(size=384)(image)
    img = img.unsqueeze(0).to(device)

    batch = {"text": [""], "image": [None]}
    inferred_token = [mp_text]
    batch["image"][0] = img


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
    heatmap_values = []
    try:
        # print(mp_text)
        # for hidx in range(1, len(encoded['input_ids'][0][:-1])):
        for token_info in token_infos:
            hidx, token, pos = token_info[0], token_info[1], token_info[2]
            selected_token = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0][hidx])
            # print(selected_token)
            if pos != 'NOUN' or selected_token != token:
                continue
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

                    # patch_index, (H, W) = infer["patch_index"]
                    heatmap = torch.zeros(H, W)
                    for i, pidx in enumerate(patch_index[0]):
                        h, w = pidx[0].item(), pidx[1].item()
                        heatmap[h, w] = cost_[i]

                    heatmap = (heatmap - heatmap.mean()) / heatmap.std()
                    heatmap_data = heatmap.detach().cpu().numpy().tolist()

                    for map_row in heatmap_data:
                        heatmap_values.extend(map_row)
    except:
        print('Error processing image')
    return heatmap_values

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

    with open('./nocaps_id_names.pkl', 'rb') as input:
        id_names = pickle.load(input)

    with open('./nocaps_visual_raw.pkl', 'rb') as input:
        visual_dict = pickle.load(input)

    domain_type = 'out-domain'

    img_ids = {}
    for id_name in id_names:
        if domain_type in id_names[id_name]:
            img_ids[id_name] = id_names[id_name]
    total_values = []
    for img_index, img_id in enumerate(img_ids):
        cap_list = visual_dict[img_id]
        for cap_index, caption_info in enumerate(cap_list):
            img_path = os.path.join('./', id_names[img_id])
            cap_id, caption, token_infos = caption_info[0], caption_info[1], caption_info[2]
            print(img_index, len(img_ids), cap_index, caption)
            single_values = compute_heat_maps(img_path, caption, token_infos, model, device, tokenizer)
            total_values.extend(single_values)


    with open('./nocaps_data/nocaps_noun_outdomain.pkl','wb') as output:
        pickle.dump(total_values, output)