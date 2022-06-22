import os
import json
from glossary import normalize_word # glossary.py is in the VLC/vlc/utils folder
import pickle


def load_label_mapping():
    file_path = '../../coco_vqa_label_info.pkl'
    with open(file_path, 'rb') as input:
        data = pickle.load(input)

    return data['ans2label'], data['label2ans']

def parse_mscoco_vqa():
    tr_file = '../../v2_mscoco_train2014_annotations.json'
    tr_data = json.load(open(tr_file, 'r'))
    tr_imgs = set()
    anns = tr_data['annotations']
    for ann_info in anns:
        tr_imgs.add(ann_info['image_id'])

    val_file = '../../v2_mscoco_val2014_annotations.json'
    val_data = json.load(open(val_file, 'r'))
    val_imgs = set()
    anns = val_data['annotations']
    for ann_info in anns:
        val_imgs.add(ann_info['image_id'])

    coco_imgs = tr_imgs.union(val_imgs)
    return coco_imgs

def get_vg_mappings():
    file_path = '../../image_data.json'
    vg_data = json.load(open(file_path, 'r'))
    vg_mapping = {}
    counter = 0
    for img_info in vg_data:
        coco_id = img_info['coco_id']
        if coco_id is not None:
            counter += 1
            vg_mapping[img_info['image_id']] = img_info['coco_id']
    return vg_mapping

def get_vgqa_data():
    file_path = '../../question_answers.json'
    vgqa_data = json.load(open(file_path, 'r'))
    vg_data = {}
    for qa_info in vgqa_data:
        qas = qa_info['qas']
        id = qa_info['id']
        assert id not in vg_data
        for qa_pair in qas:
            img_id, qa_id, question, answer = qa_pair['image_id'], qa_pair['qa_id'], qa_pair['question'], qa_pair['answer']
            if img_id not in vg_data:
                vg_data[img_id] = []
            vg_data[img_id].append((qa_id, question, answer))
    return vg_data

if __name__ == '__main__':

    coco_imgs = parse_mscoco_vqa()
    vg_mapping = get_vg_mappings()
    vgqa_data = get_vgqa_data()
    ans2label, label2ans = load_label_mapping()

    label2ans = set(label2ans)

    counter = 0
    valid_counter =0
    valid_qas = {}
    for img_id in vgqa_data:
        if img_id in vg_mapping:
            qa_list = vgqa_data[img_id]
            for (qa_id, question, answer) in qa_list:
                norm_answer = normalize_word(answer)
                if norm_answer in label2ans:
                    if img_id not in valid_qas:
                        valid_qas[img_id] = []
                    valid_qas[img_id].append((qa_id, question, norm_answer))
                    valid_counter += 1
                counter += 1
    print(counter, valid_counter)
    with open('valid_vgqa_info.pkl','wb') as output:
        pickle.dump(valid_qas, output)
