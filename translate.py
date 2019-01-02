#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse

from onmt.utils.logging import init_logger
from translate.translator import build_translator, init_fast5
from utils.labelop import index2base, simple_assembly
import numpy as np
import models.opts as opts
import multiprocessing
import os
import time
import codecs
import math


def main(opt):

    flag_intersection = False
    init_fast5(opt,math.floor(opt.sample_rate * opt.window_stride) if flag_intersection else opt.src_seq_length)

    opt.tgt = None
    opt.data_type = 'nano'
    opt.src_dir = opt.save_data

    # pool = multiprocessing.Pool(8)

    translator = build_translator(opt, report_score=True, logger=logger)

    for file_src in os.listdir(os.path.join(opt.src_dir, 'src')):

        start = time.time()

        opt.src = os.path.join(opt.save_data, 'src', file_src)
        # opt.output = os.path.join(opt.save_data, 'result', file_src)
        translator.setOutFile(codecs.open(os.path.join(opt.save_data, 'result', file_src), 'w+', 'utf-8'))

        all_scores, all_predictions = translator.translate(
            src=opt.src,
            tgt=opt.tgt,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
        )

        if flag_intersection:
            c_bpread = index2base(np.argmax(simple_assembly(all_predictions), axis=0))
        else:
            c_bpread = simple_assembly(all_predictions, flag_intersection)

        with open(os.path.join(opt.save_data, 'result', file_src.split('.')[0]+'.fasta'), 'w') as file_fasta:
            file_fasta.writelines('>%s\n%s' % (file_src.split('.')[0], c_bpread))

        end = time.time()

        with open(os.path.join(opt.save_data, 'speed.txt'), 'a+') as file_summary:
            file_summary.writelines("%s\t%0.2f\t%d\t%0.2f\n " % (
                file_src.split('.')[0], float(end - start), len(c_bpread), len(c_bpread) / float(end - start)))



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
