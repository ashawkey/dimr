import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='RfS-Net')
    parser.add_argument('--config', type=str, default='config/rfs_phase2_scannet.yaml', help='path to config file')
    parser.add_argument('--test_epoch', type=str, default='config/rfs_phase2_scannet.yaml', help='path to config file')
    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
