import os
import configargparse
import numpy as np
import random

def config_opts():
    parser = configargparse.ArgumentParser(
        description='merge_fasta.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('-src', '--source', required=True,
               help='individual fasta')
    parser.add('-o', '--output', required=True,
               help='save path')
    return parser


if __name__ == '__main__':
    parser = config_opts()
    opt = parser.parse_args()

    list_dataset = os.listdir(opt.source)

    with open(opt.output,'w') as file:
        for path_dataset in list_dataset:
            if path_dataset.endswith('fasta') or path_dataset.endswith('fastq'):
                data = open(os.path.join(opt.source,path_dataset)).read().strip()
                file.writelines(data+'\n')
