import os
import configargparse
import numpy as np
import random

def config_opts():
    parser = configargparse.ArgumentParser(
        description='rename_datasets.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('-src', '--source', required=True,
               help='preprocessed datasets')
    parser.add('-o', '--output', required=True,
               help='save path')
    return parser


if __name__ == '__main__':
    parser = config_opts()
    opt = parser.parse_args()

    list_dataset = os.listdir(opt.source)
    random.shuffle(list_dataset)

    index_dataset = 0
    for path_dataset in list_dataset:
        if 'vocab' not in path_dataset:
            os.rename(os.path.join(opt.source,path_dataset), opt.output + '.train.' + str(index_dataset) + '.pt')
            index_dataset += 1
