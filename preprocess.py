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

from utils.labelop import extract_fast5
from utils.logging import init_logger, logger

import inputters.inputter as inputters
from models import opts as opts


def init_fast5(opt, number_file_eval):
    """ extract every h5 file of src_dir into segments and save to train/eval.txt """

    # update train and valid options with files just created
    opt.train_src = os.path.join(opt.save_data, 'src-train.txt')
    opt.train_tgt = os.path.join(opt.save_data, 'tgt-train.txt')
    opt.valid_src = os.path.join(opt.save_data, 'src-eval.txt')
    opt.valid_tgt = os.path.join(opt.save_data, 'tgt-eval.txt')
    opt.data_type = 'nano'

    if os.path.exists(opt.train_src) and \
            os.path.exists(opt.train_tgt) and \
            os.path.exists(opt.valid_src) and \
            os.path.exists(opt.valid_tgt):
        opt.src_dir = opt.save_data
        return

    ind_file_train = 0
    ind_file_eval = 0

    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)

    for file_h5 in os.listdir(opt.src_dir):
        if file_h5.endswith('fast5'):

            output_prefix_feature = 'src-eval.txt' if ind_file_eval < number_file_eval else 'src-train.txt'
            output_prefix_label = 'tgt-eval.txt' if ind_file_eval < number_file_eval else 'tgt-train.txt'

            output_state = extract_fast5(os.path.join(opt.src_dir,file_h5),
                                         opt.save_data,
                                         output_prefix_feature,
                                         output_prefix_label,
                                         opt.basecall_group,
                                         opt.basecall_subgroup,
                                         opt.normalization_raw,
                                         opt.src_seq_length)
            if output_state:
                if ind_file_eval < number_file_eval:
                    ind_file_eval += 1
                else:
                    ind_file_train += 1

    opt.src_dir = opt.save_data


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
            use_filter_pred=corpus_type == 'train' or opt.filter_valid
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


def main():
    opt = parse_args()

    # assert opt.max_shard_size == 0, \
    #     "-max_shard_size is deprecated. Please use \
    #     -shard_size (number of examples) instead."
    assert opt.shuffle == 0, \
        "-shuffle is not implemented. Please shuffle \
        your data before pre-processing."

    init_fast5(opt, 10)
    # assert os.path.isfile(opt.train_src) and os.path.isfile(opt.train_tgt), \
    #     "Please check path of your train src and tgt files!"

    # assert os.path.isfile(opt.valid_src) and os.path.isfile(opt.valid_tgt), \
    #     "Please check path of your valid src and tgt files!"

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0  # tgt always text so far

    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type, src_nfeats, tgt_nfeats)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


if __name__ == "__main__":
    main()
