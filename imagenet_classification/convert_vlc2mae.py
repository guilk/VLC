import os
import torch
import argparse
from collections import OrderedDict

def get_args_parser():
    parser = argparse.ArgumentParser('Convert VLC models to MAE models', add_help=False)
    parser.add_argument('--vlc_model', default='', type=str,
                        help='the path to load pre-trained VLC model')
    parser.add_argument('--mae_model', default='', type=str, help='the folder path to save MAE model')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    vlc_model_path = args.vlc_model
    vlc_model = torch.load(vlc_model_path)
    vlc_state_dict = vlc_model['state_dict']

    mae_model = {}
    mae_state_dict = OrderedDict()
    for k,v in vlc_state_dict.items():
        if k.startswith('transformer.'):
            mae_state_dict[k[12:]] = vlc_state_dict[k]

    mae_model['model'] = mae_state_dict
    torch.save(mae_model, os.path.join(args.mae_model, 'mae_modelckpt'))