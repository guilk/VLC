import json
import pandas as pd
import pyarrow as pa
import random
import os
import pickle

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word

def path2rest(image_id, root, qa_data, ans2label):
    path = os.path.join(root, 'images/{}.jpg'.format(image_id))
    split = 'train'

    with open(path, "rb") as fp:
        binary = fp.read()

    qa_info = qa_data[image_id]
    questions = []
    answers = []
    qids = []
    answer_scores = []
    answer_labels = []
    for qa in qa_info:
        qids.append(qa[0])
        questions.append(qa[1])
        answers.append([qa[2]])
        answer_labels.append([ans2label[qa[2]]])
        answer_scores.append([1.0])
    iid = image_id

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]


def make_arrow(root, dataset_root):
    with open(os.path.join(root, 'vqa_labels_info.pkl'), 'rb') as input:
        data = pickle.load(input)

    label2ans = data['label2ans']
    ans2label = data['ans2label']

    with open(os.path.join(root, 'vg_qa_info.dict'), 'rb') as input:
        qa_data = pickle.load(input)

    bs = [
        path2rest(image_id, root, qa_data, ans2label) for image_id in tqdm(qa_data)
    ]

    dataframe = pd.DataFrame(
        bs,
        columns=[
            "image",
            "questions",
            "answers",
            "answer_labels",
            "answer_scores",
            "image_id",
            "question_id",
            "split",
        ],
    )

    table = pa.Table.from_pandas(dataframe)

    os.makedirs(dataset_root, exist_ok=True)
    with pa.OSFile(f"{dataset_root}/vgqa_train.arrow", "wb") as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)