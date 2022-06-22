import os
from PIL import Image
import multiprocessing as mp

import warnings
warnings.filterwarnings("error")

def worker(img_path):

    try:
        img = pil_loader(img_path)
    except:
        print('Error loading {}'.format(img_path))
        cmd = 'rm {}'.format(img_path)
        os.system(cmd)

def pil_loader(img_path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def clean_gcc():
    img_root = 'ROOT/GCC/images_train/'

    img_paths = []
    img_folders = os.listdir(img_root)
    counter = 0
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            counter += 1
            img_path = os.path.join(folder_path, img_name)
            img_paths.append((img_path,))

    before_num = counter
    pool = mp.Pool(90)

    pool.starmap(worker, img_paths)

    img_paths = []
    img_folders = os.listdir(img_root)
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_paths.append(img_path)
    after_num = len(img_paths)
    print('before images: {}; after images: {}'.format(before_num, after_num))

def clean_sbu():
    img_root = 'ROOT/SBU/images_train/'
    img_paths = []
    img_folders = os.listdir(img_root)
    # img_folders = img_folders[:1]
    counter = 0
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            counter += 1
            img_path = os.path.join(folder_path, img_name)
            img_paths.append((img_path,))

    before_num = counter
    pool = mp.Pool(90)

    pool.starmap(worker, img_paths)

    img_paths = []
    img_folders = os.listdir(img_root)
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_paths.append(img_path)
    after_num = len(img_paths)
    print('before images: {}; after images: {}'.format(before_num, after_num))

def clean_vg():
    img_root = 'ROOT/VG/images/'
    img_paths = []
    img_folders = os.listdir(img_root)
    # img_folders = img_folders[:1]
    counter = 0
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            counter += 1
            img_path = os.path.join(folder_path, img_name)
            img_paths.append((img_path,))

    before_num = counter
    pool = mp.Pool(90)

    pool.starmap(worker, img_paths)

    img_paths = []
    img_folders = os.listdir(img_root)
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_paths.append(img_path)
    after_num = len(img_paths)
    print('before images: {}; after images: {}'.format(before_num, after_num))

def clean_coco():
    img_root = 'ROOT/COCO/'
    img_folders = ['train2014', 'val2014']
    counter = 0
    img_paths = []

    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            counter += 1
            img_path = os.path.join(folder_path, img_name)
            img_paths.append((img_path,))

    before_num = counter
    pool = mp.Pool(90)

    pool.starmap(worker, img_paths)

    img_paths = []
    img_folders = ['train2014', 'val2014']
    for img_folder in img_folders:
        folder_path = os.path.join(img_root, img_folder)
        img_names = os.listdir(folder_path)
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img_paths.append(img_path)
    after_num = len(img_paths)
    print('before images: {}; after images: {}'.format(before_num, after_num))

if __name__ == '__main__':
    clean_gcc()
    clean_sbu()
    clean_vg()
    clean_coco()