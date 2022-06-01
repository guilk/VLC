from sklearn.cluster import KMeans
import numpy as np
import pickle
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def get_color_dict(labels):
    color_dict = {}
    for label in labels:
        if label not in color_dict:
            color_mask = np.random.random((1, 3)).tolist()[0]
            color_dict[label] = color_mask
    return color_dict

def visualize_ours():
    seed = 1024
    np.random.seed(seed)

    features = pickle.load(open('ours_livingroom_feats.pkl','rb'))
    # features = preprocessing.normalize(features)
    n_clusters = 8
    model = KMeans(n_clusters=n_clusters).fit(features.astype(np.float))
    clusters = model.predict(features.astype(np.float))

    labels = np.unique(clusters)
    color_dict = get_color_dict(labels)

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

    cluster_map = torch.zeros(H, W)
    for i, pidx in enumerate(patch_index[0]):
        h, w = pidx[0].item(), pidx[1].item()
        cluster_map[h,w] = clusters[i]
    cluster_map = cluster_map.numpy()

    dst_img = np.zeros((H*16,W*16,3))

    for i in range(0, H):
        for j in range(0, W):
            for k in range(3):
                dst_img[i*16:(i+1)*16,j*16:(j+1)*16,k] = color_dict[cluster_map[i,j]][k]

    img_path = './livingroom.jpeg'
    image = Image.open(img_path).convert("RGB")
    _w, _h = image.size
    save_img = Image.fromarray(np.uint8(dst_img * 255), "RGB").resize(
                        (_w, _h), resample=Image.NEAREST)

    save_img.save('visualized_result.jpg')


if __name__ == '__main__':
    # different seed can generate different color dict that will affect the visiblity of clustering results
    # This code is to reproduce Figure 3 in our paper
    # Before running it, you need to run 'extract_clustering_features.py' first
    seed = 1024
    np.random.seed(seed)
    visualize_ours()