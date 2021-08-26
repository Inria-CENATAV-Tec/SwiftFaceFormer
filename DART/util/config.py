""" Config class for search/augment """
import argparse
import os
from models import genotypes as gt
from functools import partial
import torch

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]

# Base class to store and print configurations
class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        """ prints configs """
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Returns configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)
        
        return text

# Class to store config data of search
class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', default='CASIA',help='CASIA')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size') # standard 64
        parser.add_argument('--w_lr', type=float, default=0.1, help='lr for weights') # 0.1 / 0.21 / 0.025
        parser.add_argument('--w_lr_min', type=float, default=0.004, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4, help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5., help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
        parser.add_argument('--gpus', default='all', help='gpu device ids seperated by comma. "all" indicates use all gpu.')
        parser.add_argument('--epochs', type=int, default=10, help='# of training epochs.')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int, default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--workers', type=int, default=4, help='# of workers')
        parser.add_argument('--alpha_lr', type=float, default=12e-4, help='lr for alpha')                     # 0.1 / 0.21
        parser.add_argument('--alpha_weight_decay', type=float, default=1e-3, help='weight decay for alpha')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join('searchs_output', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)