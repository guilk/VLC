import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn import preprocessing


def get_vilt_data(file_path):
    with open(file_path,'rb') as input:
        data = pickle.load(input)
    return np.asarray(data)

def get_ours_data(file_path):
    with open(file_path,'rb') as input:
        data = pickle.load(input)
    return np.asarray(data)

def min_max_normalize(data_list):
    min_value = min(data_list)
    max_value = max(data_list)
    norm_list = [(value - min_value)/(max_value-min_value) for value in data_list]
    return norm_list

if __name__ == '__main__':
    token_type = 'noun'
    domain_type = 'outdomain'
    norm_type = 'unnorm'
    scaler = preprocessing.MinMaxScaler()


    vilt_path = '/home/liangkeg/internship/mim/visualizations/nocaps_data/nocaps_vilt_{}_{}_{}.pkl'.format(token_type, domain_type, norm_type)
    vilt_values = get_vilt_data(vilt_path)
    vilt_values = min_max_normalize(vilt_values)

    ours_path = '/home/liangkeg/internship/mim/visualizations/nocaps_data/nocaps_ours_{}_{}_{}.pkl'.format(token_type, domain_type, norm_type)
    ours_values = get_ours_data(ours_path)
    ours_values = min_max_normalize(ours_values)

    if norm_type == 'norm':
        bins = np.linspace(-3, 3, 50)
    else:
        bins = np.linspace(0, 0.5, 50)
    plt.hist(vilt_values, bins, density = True, alpha=0.5, label='vilt')
    plt.hist(ours_values, bins, density = True, alpha=0.5, label='ours')
    plt.legend(loc='upper right')
    plt.gca().set(title='{} {} Probability Density'.format(domain_type, token_type), xlabel='Token-patch Similarity Scores', ylabel='Probability Density')
    plt.show()
    plt.savefig('./{}_{}_{}.jpg'.format(domain_type, token_type, norm_type))

