#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse

from onmt.utils.logging import init_logger
from translate.translator import build_translator
from utils.labelop import index2base, simple_assembly, extract_fast5_raw
import numpy as np
import models.opts as opts
import multiprocessing
import os
import time
import codecs

def main(opt):

    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)

    if not os.path.exists(os.path.join(opt.save_data, 'src')):
        os.makedirs(os.path.join(opt.save_data, 'src'))

    if not os.path.exists(os.path.join(opt.save_data, 'result')):
        os.makedirs(os.path.join(opt.save_data, 'result'))

    if not os.path.exists(os.path.join(opt.save_data, 'segment')):
        os.makedirs(os.path.join(opt.save_data, 'segment'))

    if not os.path.exists(os.path.join(opt.save_data, 'attention')):
        os.makedirs(os.path.join(opt.save_data, 'attention'))

    opt.tgt = None
    opt.data_type = 'nano'

    translator = build_translator(opt, report_score=True, logger=logger)

    def translate(file_src):

        start = time.time()

        opt.src = os.path.join(opt.save_data, 'src', file_src)
        # opt.output = os.path.join(opt.save_data, 'result', file_src)
        translator.setOutFile(codecs.open(os.path.join(opt.save_data, 'segment', file_src), 'w+', 'utf-8'))

        if opt.attn_debug:
            translator.setAttnFile(codecs.open(os.path.join(opt.save_data, 'attention', file_src), 'a+', 'utf-8'))

        all_scores, all_predictions = translator.translate(
            src=opt.src,
            tgt=opt.tgt,
            src_dir=opt.save_data,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
        )

        if opt.src_seq_stride < opt.src_seq_length:
            c_bpread = index2base(np.argmax(simple_assembly(all_predictions), axis=0))
        else:
            c_bpread = simple_assembly(all_predictions, flag_intersection=False)

        with open(os.path.join(opt.save_data, 'result', file_src.split('.')[0] + '.fasta'), 'w') as file_fasta:
            file_fasta.writelines('>%s\n%s' % (file_src.split('.')[0], c_bpread))

        end = time.time()

        with open(os.path.join(opt.save_data, 'speed.txt'), 'a+') as file_summary:
            file_summary.writelines("%s\t%0.2f\t%d\t%0.2f\n " % (
                file_src.split('.')[0], float(end - start), len(c_bpread), len(c_bpread) / float(end - start)))

    pool = multiprocessing.Pool(opt.thread)

    for file_h5 in os.listdir(opt.src_dir):
        if file_h5.endswith('fast5'):
            output_prefix_feature = file_h5.split('.fast5')[0] + '.txt'

            pool.apply_async(extract_fast5_raw,
                             (os.path.join(opt.src_dir, file_h5),
                              opt.save_data,
                              output_prefix_feature,
                              opt.normalization_raw,
                              opt.src_seq_length,
                              opt.src_seq_stride,),
                             callback=translate)

    pool.close()
    pool.join()


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
