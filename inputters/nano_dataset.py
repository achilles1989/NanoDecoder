# -*- coding: utf-8 -*-
"""
    NanoDataset
"""
import codecs
import os
import sys
import io
import struct
from tqdm import tqdm
from statsmodels import robust

import torch
import torchtext
from onmt.inputters.text_dataset import TextDataset

from onmt.inputters.dataset_base import DatasetBase, PAD_WORD, BOS_WORD, \
    EOS_WORD

DNA_IDX = ['A', 'C', 'G', 'T']


class NanoDataset(DatasetBase):
    """ Dataset for data_type=='dna'

        Build `Example` objects, `Field` objects, and filter_pred function
        from h5 corpus.

        Args:
            fields (dict): a dictionary of `torchtext.data.Field`.
            src_examples_iter (dict iter): preprocessed source example
                dictionary iterator.
            tgt_examples_iter (dict iter): preprocessed target example
                dictionary iterator.
            tgt_seq_length (int): maximum target sequence length.
            use_filter_pred (bool): use a custom filter predicate to filter
                out examples?
    """

    def sort_key(self, ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def extract_features(audio_path, sample_rate, truncate, flag_fft, window_size,
                         window_stride, window, normalize_audio):
        global librosa, np
        import librosa
        import numpy as np

        f_h = open(audio_path, 'r')
        signal = list()
        for line in f_h:
            signal += [float(x) for x in line.split()]
        signal = np.asarray(signal)

        if truncate and truncate > 0:
            if signal.size(0) > truncate:
                signal = signal[:truncate]

        if flag_fft:
            n_fft = int(sample_rate * window_size)
            win_length = n_fft
            hop_length = int(sample_rate * window_stride)
            # STFT
            d = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=window)
            spect, _ = librosa.magphase(d)
            spect = np.log1p(spect)
            spect = torch.FloatTensor(spect)
            if normalize_audio:
                mean = spect.mean()
                std = spect.std()
                spect.add_(-mean)
                spect.div_(std)
        else:

            spect = torch.FloatTensor(signal).view(1,-1)

        return spect

    @classmethod
    def make_examples(
            cls,
            path,
            src_dir,
            side,
            flag_fft,
            sample_rate,
            window_size,
            window_stride,
            window,
            normalize_audio,
            truncate=None
    ):
        """
        Args:
            path (str): location of a src file containing h5 segments path.
            src_dir (str): location of source segments files.
            side (str): 'src' or 'tgt'.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Yields:
            a dictionary containing audio data for each line.
        """
        assert isinstance(path, str), "Iterators not supported for audio"
        assert src_dir is not None and os.path.exists(src_dir), \
            "src_dir must be a valid directory if data_type is audio"

        with codecs.open(path, "r", "utf-8") as corpus_file:
            for i, line in enumerate(tqdm(corpus_file)):
                audio_path = os.path.join(src_dir, line.strip())
                if not os.path.exists(audio_path):
                    audio_path = line.strip()

                assert os.path.exists(audio_path), \
                    'audio path %s not found' % (line.strip())

                spect = NanoDataset.extract_features(
                    audio_path, sample_rate, truncate, flag_fft, window_size,
                    window_stride, window, normalize_audio
                )

                yield {side: spect, side + '_path': line.strip(),
                       side + '_lengths': spect.size(1), 'indices': i}