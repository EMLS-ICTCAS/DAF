import json
import argparse
from trainer import train
import copy

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    args1 = copy.deepcopy(args)
    args.update(param) # Add parameters from json
    
    args.update(args1)

    train(args)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Dynamical Adapter Fusion (DAF) for PTM-based class-incremental learning.')
    parser.add_argument('--config', type=str, default='./exps/daf_cifar.json',
                        help='Json file of settings.')
    
    return parser

if __name__ == '__main__':
    main()
