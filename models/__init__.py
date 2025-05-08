from models.mynet.mynet import EAG_SAE_ESTF

from common import config
import argparse
import torch

def create_model(args) -> torch.nn.Module:

    archs = [EAG_SAE_ESTF]
    archs_dict = {a.__name__.lower(): a for a in archs} 
    arch = args['architecture'] 
    try:
        model_class = archs_dict[arch.lower()] 
        print("model_class",model_class)
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
            arch, list(archs_dict.keys()),
        ))

    if arch.lower() == arch: 
        return model_class()
    else:
        raise RuntimeError('No implementation: ', arch.lower())

def get_parser():
    parser = argparse.ArgumentParser(description='--')
    parser.add_argument('--config', type=str, default='/config/CHUAC.yaml', help='Model config file') #
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

