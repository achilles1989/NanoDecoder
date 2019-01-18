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
import sys


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
    def log(self,s):
        sys.stdout.write(' ' * (self.width + 20) + '\r')
        sys.stdout.flush()
        if self.total == 0: return
        # print(s)
        progress = int(self.width * self.count / self.total)
        # print(self.width,self.count,progress)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress))
        if self.count > 0:
            s_estimate = s * self.total / self.count
        else:
            s_estimate = 0

        m, s = divmod(s, 60)
        h, m = divmod(m, 60)

        sys.stdout.write(' %02d:%02d:%02d' % (h, m, s))

        m, s = divmod(s_estimate,60)
        h, m = divmod(m, 60)

        sys.stdout.write(' / %02d:%02d:%02d' % (h, m, s) + '\r')

        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


def main(opt):

    if not os.path.exists(opt.save_data):
        os.mkdir(opt.save_data)

    if not os.path.exists(os.path.join(opt.save_data, 'result')):
        os.makedirs(os.path.join(opt.save_data, 'result'))

    if not os.path.exists(os.path.join(opt.save_data, 'segment')):
        os.makedirs(os.path.join(opt.save_data, 'segment'))

    if (not os.path.exists(os.path.join(opt.save_data, 'attention'))) and opt.attn_debug:
        os.makedirs(os.path.join(opt.save_data, 'attention'))

    opt.tgt = None
    opt.data_type = 'nano'

    translator = build_translator(opt, report_score=False, logger=logger)
    pool = multiprocessing.Pool(opt.thread)
    bar = ProgressBar()
    start_task = time.time()

    def writeOutPut(file_src, all_predictions, time_translate, time_task):

        if opt.src_seq_stride < opt.src_seq_length:
            c_bpread = index2base(np.argmax(simple_assembly(all_predictions), axis=0))
        else:
            c_bpread = simple_assembly(all_predictions, flag_intersection=False)
        with open(os.path.join(opt.save_data, 'result', file_src.split('.')[0] + '.fasta'), 'w') as file_fasta:
            file_fasta.writelines('>%s\n%s' % (file_src.split('.')[0], c_bpread))
        with open(os.path.join(opt.save_data, 'segment', file_src), 'w+') as file_segment:
            for n_best_preds in all_predictions:
                file_segment.write('\n'.join(n_best_preds) + '\n')
        with open(os.path.join(opt.save_data, 'speed.txt'), 'a+') as file_summary:
            file_summary.writelines("%s\t%0.2f\t%d\t%0.2f\n" % (
                file_src.split('.')[0], float(time_translate), len(c_bpread), len(c_bpread) / float(time_translate)))

        bar.log(time_task)

    def translate(source_data):

        if (not source_data) or len(source_data) == 1:
            return

        file_src = source_data[0]
        src_data = source_data[1:]

        start = time.time()

        if opt.attn_debug:
            translator.setAttnFile(codecs.open(os.path.join(opt.save_data, 'attention', file_src), 'a+', 'utf-8'))

        all_scores, all_predictions = translator.translate(
            # src=opt.src,
            src=src_data,
            tgt=opt.tgt,
            src_dir=opt.save_data,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
        )

        end = time.time()

        multiprocessing.Process(target=writeOutPut,args=(file_src,
                          all_predictions,
                          end-start,
                          end-start_task,)).start()

        bar.move()

    total_h5 = 0
    total_h5_translated = 0

    num_limited = 500

    for file_h5 in os.listdir(opt.src_dir):
        if file_h5.endswith('fast5'):
            output_prefix_feature = file_h5.split('.fast5')[0] + '.txt'
            flag_extract = 'fast5'
        if file_h5.endswith('signal'):
            output_prefix_feature = file_h5.split('.signal')[0] + '.txt'
            flag_extract = 'signal'

            # pool.apply_async(extract_fast5_raw,
            #                  (os.path.join(opt.src_dir, file_h5),
            #                   opt.save_data,
            #                   output_prefix_feature,
            #                   opt.normalization_raw,
            #                   opt.src_seq_length,
            #                   opt.src_seq_stride,),
            #                  callback=translate)
        if not os.path.exists(os.path.join(opt.save_data, 'result', output_prefix_feature.split('.')[0]+'.fasta')):
            total_h5 += 1
            pool.apply_async(extract_fast5_raw,
                             (os.path.join(opt.src_dir, file_h5),
                              output_prefix_feature,
                              opt.normalization_raw,
                              opt.src_seq_length,
                              opt.src_seq_stride,
                              flag_extract,),
                             callback=translate)
        else:
            total_h5_translated += 1

        if total_h5 + total_h5_translated >= num_limited:
            break

    logger.info('%d reads have already translated, remains %d read\n' % (total_h5_translated, total_h5))
    bar.setTotal(total_h5)
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
