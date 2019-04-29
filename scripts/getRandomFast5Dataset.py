import os
import subprocess
import random
import configargparse


def config_opts():
    parser = configargparse.ArgumentParser(
        description='merge_fasta.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('-src', '--source', required=True,
               help='raw fast5')
    parser.add('-o', '--output', required=True,
               help='save path')
    parser.add('-n', '--number', required=True,
               help='number to select')
    return parser


if __name__ == '__main__':

    parser = config_opts()
    opt = parser.parse_args()

    samples = random.sample(os.listdir(opt.source), int(opt.number))

    for sample in samples:
        filename = os.path.join(opt.source, sample)
        command_cp = 'cp ' + filename + ' ' + os.path.join(opt.save, '01_raw_fast5')
        subprocess.call(command_cp, shell=True)

    command_ls = 'ls ' + os.path.join(opt.save, '01_raw_fast5') + ' > ' + os.path.join(opt.save, 'list_fast5.txt')
    subprocess.call(command_ls, shell=True)
