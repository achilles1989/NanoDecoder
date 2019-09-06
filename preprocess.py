#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import configargparse
import glob
import sys
import gc
import os
import codecs
import torch
import multiprocessing
import time
import math
import shutil
import random

from utils.labelop import extract_fast5, extract_signal
from utils.logging import init_logger, logger

import inputters.inputter as inputters
from models import opts as opts


class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    # def log(self, s):
    def setTotal(self,total):
        self.total = total
    def log(self):
        sys.stdout.write(' ' * (self.width + 20) + '\r')
        sys.stdout.flush()
        if self.total == 0: return
        # print(s)
        progress = int(self.width * self.count / self.total)
        # print(self.width,self.count,progress)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress))

        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

def init_fast5(opt,group_src):
    """ extract every h5 file of src_dir into segments and save to train/eval.txt """

    # update train and valid options with files just created

    opt.data_type = 'nano'

    opt.ind_file_train = 0
    opt.ind_file_eval = 0
    src_dir = opt.src_dir
    opt.src_dir = opt.save_data
    prefix = opt.prefix
    opt.index_group = 0

    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)

    bar = ProgressBar()
    # start_task = time.time()

    opt.train_src = os.path.join(opt.save_data, 'src-train.txt')
    opt.train_tgt = os.path.join(opt.save_data, 'tgt-train.txt')
    opt.valid_src = os.path.join(opt.save_data, 'src-eval.txt')
    opt.valid_tgt = os.path.join(opt.save_data, 'tgt-eval.txt')

    removeTEMP(opt, [])

    dict_all_label = {}

    def generateDatasets(tmp_len):

        if opt.ind_file_eval < opt.number_file_eval:
            opt.ind_file_eval += 1
        else:
            opt.ind_file_train += 1
        output_prefix_feature = 'src-eval.txt' if opt.ind_file_eval < opt.number_file_eval else 'src-train.txt'
        output_prefix_label = 'tgt-eval.txt' if opt.ind_file_eval < opt.number_file_eval else 'tgt-train.txt'
        with open(os.path.join(opt.save_data, output_prefix_feature), 'a+') as file_output_feature_summary, open(
                os.path.join(opt.save_data, output_prefix_label), 'a+') as file_output_label:
            list_rm = []
            tmp_keys = list(dict_all_label.keys())
            random.shuffle(tmp_keys)
            while len(dict_all_label) > tmp_len - opt.shard_size:
                if not len(dict_all_label):
                    break
                del_key = tmp_keys.pop(0)
                tuple_value = dict_all_label.pop(del_key)
                file_output_feature_summary.writelines(del_key + '\n')
                file_output_label.writelines(tuple_value + '\n')
                list_rm.append(del_key)

        opt.prefix = prefix + '.' + str(opt.index_group)

        logger.info("Extracting features...")

        src_nfeats = 0
        tgt_nfeats = 0  # tgt always text so far

        # logger.info(" * number of source features: %d." % src_nfeats)
        # logger.info(" * number of target features: %d." % tgt_nfeats)

        logger.info("Building `Fields` object...")
        fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

        logger.info("Building & saving training data...")
        train_dataset_files = build_save_dataset('train', fields, opt)

        if opt.number_file_eval > 0:
            logger.info("Building & saving validation data...")
            build_save_dataset('valid', fields, opt)

        logger.info("Building & saving vocabulary...")
        build_save_vocab(train_dataset_files, fields, opt)

        opt.index_group += 1

        removeTEMP(opt, list_rm)

    def writeToFile(dict_label):

        dict_all_label.update(dict_label)
        bar.move()
        tmp_len = len(dict_all_label)
        if tmp_len > opt.shard_size:
            generateDatasets(tmp_len)

        bar.log()
        return True

    total_h5 = 0
    min_label = 0

    pool = multiprocessing.Pool(opt.thread)

    for file_h5 in group_src:
        if file_h5.endswith('fast5'):
            total_h5 += 1
            pool.apply_async(extract_fast5,
                             (os.path.join(src_dir, file_h5),
                             opt.save_data,
                             opt.basecall_group,
                             opt.basecall_subgroup,
                             opt.normalization_raw,
                             opt.src_seq_length,
                             opt.cpg,
                             min_label,
                              ),
                             callback=writeToFile)
        if file_h5.endswith('signal'):
            total_h5 += 1
            pool.apply_async(extract_signal,
                             (os.path.join(src_dir, file_h5),
                             opt.save_data,
                             opt.basecall_group,
                             opt.basecall_subgroup,
                             opt.normalization_raw,
                             opt.src_seq_length,
                             opt.cpg,
                             min_label,
                              ),
                             callback=writeToFile)

    bar.setTotal(total_h5)
    pool.close()
    pool.join()

    if len(dict_all_label) > 0:
        generateDatasets(len(dict_all_label))
    opt.prefix = prefix

def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '/'+opt.prefix+'.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def parse_args():
    parser = configargparse.ArgumentParser(
        description='preprocess.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    check_existing_pt_files(opt)

    return opt


def _write_shard(path, data, start, end=None):
    with codecs.open(path, "w", encoding="utf-8") as f:
        shard = data[start:end] if end is not None else data[start:]
        f.writelines(shard)


def _write_temp_shard_files(corpus, fields, corpus_type, shard_size):
    # Does this actually shard in a memory-efficient way? The readlines()
    # reads in the whole corpus. Shards should be efficient at training time,
    # but in principle it should not be necessary to read everything at once
    # when preprocessing either.
    with codecs.open(corpus, "r", encoding="utf-8") as f:
        data = f.readlines()
        corpus_size = len(data)

    if shard_size <= 0:
        shard_size = corpus_size
    for i, start in enumerate(range(0, corpus_size, shard_size)):
        logger.info("Splitting shard %d." % i)
        end = start + shard_size
        shard_path = corpus + ".{}.txt".format(i)
        _write_shard(shard_path, data, start, end)

    return corpus_size


def build_save_dataset(corpus_type, fields, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        tgt = opt.train_tgt
    else:
        src = opt.valid_src
        tgt = opt.valid_tgt

    logger.info("Reading source and target files: %s %s." % (src, tgt))
    src_len = _write_temp_shard_files(src, fields, corpus_type, opt.shard_size)
    tgt_len = _write_temp_shard_files(tgt, fields, corpus_type, opt.shard_size)
    assert src_len == tgt_len, "Source and target should be the same length"

    src_shards = sorted(glob.glob(src + '.*.txt'))
    tgt_shards = sorted(glob.glob(tgt + '.*.txt'))
    shard_pairs = zip(src_shards, tgt_shards)
    dataset_paths = []

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Building shard %d." % i)
        dataset = inputters.build_dataset(
            fields, opt.data_type,
            src=src_shard,
            tgt=tgt_shard,
            src_dir=opt.src_dir,
            src_seq_len=opt.src_seq_length,
            tgt_seq_len=opt.tgt_seq_length,
            src_seq_length_trunc=opt.src_seq_length_trunc,
            tgt_seq_length_trunc=opt.tgt_seq_length_trunc,
            # dynamic_dict=opt.dynamic_dict,
            flag_fft=opt.fft,
            sample_rate=opt.sample_rate,
            window_size=opt.window_size,
            window_stride=opt.window_stride,
            window=opt.window,
            # image_channel_size=opt.image_channel_size,
            use_filter_pred=corpus_type == 'train' or opt.filter_valid,
            corpus_type=corpus_type
        )

        # data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        data_path = "{:s}/{:s}.{:s}.{:d}.pt".format(opt.save_data, opt.prefix, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        os.remove(src_shard)
        os.remove(tgt_shard)
        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type
        # opt.share_vocab,
        # opt.src_vocab,
        # opt.src_vocab_size,
        # opt.src_words_min_frequency,
        # opt.tgt_vocab,
        # opt.tgt_vocab_size,
        # opt.tgt_words_min_frequency
    )

    vocab_path = opt.save_data + '/'+opt.prefix+'.vocab.pt'
    torch.save(inputters.save_fields_to_vocab(fields), vocab_path)


def removeTEMP(opt, list_rm):

    if len(list_rm) > 0:
        for rm_values in list_rm:
            os.remove(os.path.join(opt.save_data,rm_values))
    else:
        if os.path.exists(os.path.join(opt.save_data, 'segments')):
            shutil.rmtree(os.path.join(opt.save_data, 'segments'))

    if os.path.exists(opt.train_src):
        os.remove(opt.train_src)
    if os.path.exists(opt.train_tgt):
        os.remove(opt.train_tgt)
    if os.path.exists(opt.valid_src):
        os.remove(opt.valid_src)
    if os.path.exists(opt.valid_tgt):
        os.remove(opt.valid_tgt)


def shuffle(opt):
    list_dataset = os.listdir(opt.save_data)
    random.shuffle(list_dataset)
    index_train = 0
    index_valid = 0
    for path_dataset in list_dataset:

        if opt.prefix in path_dataset:
            if 'vocab' not in path_dataset:
                if 'train' in path_dataset:
                    os.rename(os.path.join(opt.save_data,path_dataset), os.path.join(opt.save_data, opt.prefix+'.train.' + str(index_train) + '.pt'))
                    index_train += 1
                if 'valid' in path_dataset:
                    os.rename(os.path.join(opt.save_data,path_dataset), os.path.join(opt.save_data, opt.prefix + '.valid.' + str(index_valid) + '.pt'))
                    index_valid += 1
            elif os.path.exists(os.path.join(opt.save_data, opt.prefix + '.vocab.pt')):
                os.remove(os.path.join(opt.save_data, path_dataset))
            else:
                os.rename(os.path.join(opt.save_data, path_dataset),
                          os.path.join(opt.save_data, opt.prefix + '.vocab.pt'))


def main():
    opt = parse_args()

    # assert opt.max_shard_size == 0, \
    #     "-max_shard_size is deprecated. Please use \
    #     -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle \
        your data before pre-processing."

    opt.number_file_eval = 0

    data_src = os.listdir(opt.src_dir)
    random.shuffle(data_src)
    len_data = len(data_src)

    # group_read = 800
    # prefix = opt.prefix
    init_logger(opt.log_file)

    # opt.train_src = os.path.join(opt.save_data, 'src-train.txt')
    # opt.train_tgt = os.path.join(opt.save_data, 'tgt-train.txt')
    # opt.valid_src = os.path.join(opt.save_data, 'src-eval.txt')
    # opt.valid_tgt = os.path.join(opt.save_data, 'tgt-eval.txt')

    # src_dir = opt.src_dir

    # for index_group in range(math.ceil(len_data/group_read)):

    # opt.src_dir = src_dir

    # group_src = data_src[index_group*group_read:(index_group+1)*group_read if (index_group+1)*group_read < len_data else len_data]
    # opt.prefix = prefix + '.' + str(index_group)

    init_fast5(opt,data_src)
    # assert os.path.isfile(opt.train_src) and os.path.isfile(opt.train_tgt), \
    #     "Please check path of your train src and tgt files!"

    # assert os.path.isfile(opt.valid_src) and os.path.isfile(opt.valid_tgt), \
    #     "Please check path of your valid src and tgt files!"


    # logger.info("Extracting features...")
    #
    # src_nfeats = 0
    # tgt_nfeats = 0  # tgt always text so far
    #
    # logger.info(" * number of source features: %d." % src_nfeats)
    # logger.info(" * number of target features: %d." % tgt_nfeats)
    #
    # logger.info("Building `Fields` object...")
    # fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)
    #
    # logger.info("Building & saving training data...")
    # train_dataset_files = build_save_dataset('train', fields, opt)
    #
    # if opt.number_file_eval > 0:
    #     logger.info("Building & saving validation data...")
    #     build_save_dataset('valid', fields, opt)

    # opt.prefix = prefix
    # logger.info("Building & saving vocabulary...")
    # build_save_vocab(train_dataset_files, fields, opt)
    shuffle(opt)


if __name__ == "__main__":
    main()
