# -*- coding: utf-8 -*-
"""
    DNADataset
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

class DNADataset(DatasetBase):
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

    def __init__(self, fields, src_examples_iter,flag_translate=None,
                 tgt_seq_length=0, use_filter_pred=True):
        self.data_type = 'dna'
        self.n_src_feats = 0
        self.n_tgt_feats = 0

        if flag_translate != None:
            examples_iter = (self._join_dicts(src, tgt) for src, tgt in
                          src_examples_iter)
        else:
            examples_iter = src_examples_iter

        # Peek at the first to see which fields are used.
        ex, examples_iter = self._peek(examples_iter)
        keys = ex.keys()

        out_fields = [(k, fields[k]) if k in fields else (k, None)
                      for k in keys]
        example_values = ([ex[k] for k in keys] for ex in examples_iter)
        out_examples = (self._construct_example_fromlist(
            ex_values, out_fields)
            for ex_values in example_values)
        # If out_examples is a generator, we need to save the filter_pred
        # function in serialization too, which would cause a problem when
        # `torch.save()`. Thus we materialize it as a list.
        out_examples = list(out_examples)

        def filter_pred(example):
            """    ?    """
            # if tgt_examples_iter is not None:
            #     return 0 < len(example.tgt) <= tgt_seq_length
            # else:
            return True

        filter_pred = filter_pred if use_filter_pred else lambda x: True

        super(DNADataset, self).__init__(
            out_examples, out_fields, filter_pred
        )

    def sort_key(self, ex):
        """ Sort using duration time of the sound spectrogram. """
        return ex.src.size(1)

    @staticmethod
    def make_audio_examples_nfeats_tpl(path, audio_dir,
                                       sample_rate, window_size,
                                       window_stride, window,
                                       normalize_audio, truncate=None):
        """
        Args:
            path (str): location of a src file containing audio paths.
            audio_dir (str): location of source audio files.
            sample_rate (int): sample_rate.
            window_size (float) : window size for spectrogram in seconds.
            window_stride (float): window stride for spectrogram in seconds.
            window (str): window type for spectrogram generation.
            normalize_audio (bool): subtract spectrogram by mean and divide
                by std or not.
            truncate (int): maximum audio length (0 or None for unlimited).

        Returns:
            (example_dict iterator, num_feats) tuple
        """
        examples_data_iter = DNADataset.read_audio_file(
            path, audio_dir,sample_rate,
            window_size, window_stride, window,
            normalize_audio, truncate)
        num_feats = 0  # Source side(audio) has no features.

        return examples_data_iter, num_feats

    @staticmethod
    def extract_features(audio_path, sample_rate, truncate, window_size,
                         window_stride, window, normalize_audio):
        global torchaudio, librosa, np
        import torchaudio
        import librosa
        import numpy as np

        def read_signal(file_path, normalize="median"):
            f_h = open(file_path, 'r')
            signal = list()
            for line in f_h:
                signal += [float(x) for x in line.split()]
            signal = np.asarray(signal)
            if len(signal) == 0:
                return signal.tolist()
            if normalize == "mean":
                signal = (signal - np.mean(signal)) / np.float(np.std(signal))
            elif normalize == "median":
                signal = (signal - np.median(signal)) / np.float(robust.mad(signal))
            return signal.tolist()

        def padding(x, L, padding_list=None):
            """Padding the vector x to length L"""
            len_x = len(x)
            assert len_x <= L, "Length of vector x is larger than the padding length"
            zero_n = L - len_x
            if padding_list is None:
                x.extend([0] * zero_n)
            elif len(padding_list) < zero_n:
                x.extend(padding_list + [0] * (zero_n - len(padding_list)))
            else:
                x.extend(padding_list[0:zero_n])
            return None

        ind_data = 0
        ind_label = 513

        step = 30
        seg_length = 300

        datalist = []
        wordlist = []
        featslist = []

        if audio_path.endswith('.bin'):

            with open(audio_path, mode='rb') as file:
                fileContent = file.read()

            for group_data in range(1, int(len(fileContent) / 2564) + 1):
                tmpdata = struct.unpack("<1H512f1H512b", fileContent[((group_data - 1) * 2564):group_data * 2564])
                length = tmpdata[ind_data]
                # print(tmpdata[ind_data+1:ind_label-1])
                label_length = tmpdata[ind_label]
                # print(tmpdata[ind_label+1:ind_label + 1 + label_length])
                raw_data = tmpdata[ind_data + 1:length]
                tmp_label = tmpdata[ind_label + 1:ind_label + 1 + label_length]
                raw_label = [DNA_IDX[x] for x in tmp_label]
                words, feats, n_feats = TextDataset.extract_text_features(raw_label)

                datalist.append(raw_data)
                wordlist.append(words)
                featslist.append(feats)

            return datalist, wordlist, featslist, group_data

        elif audio_path.endswith('.signal'):

            event = list()
            event_len = list()
            f_signal = read_signal(audio_path, normalize='median')
            sig_len = len(f_signal)
            group_data = 0
            for indx in range(0, sig_len, step):
                segment_sig = f_signal[indx:indx + seg_length]
                segment_len = len(segment_sig)
                padding(segment_sig, seg_length)
                event.append(segment_sig)
                event_len.append(segment_len)
                group_data += 1

            return event,group_data


        # print(group_data)


        # sound, sample_rate_ = torchaudio.load(audio_path)
        # if truncate and truncate > 0:
        #     if sound.size(0) > truncate:
        #         sound = sound[:truncate]
        #
        # assert sample_rate_ == sample_rate, \
        #     'Sample rate of %s != -sample_rate (%d vs %d)' \
        #     % (audio_path, sample_rate_, sample_rate)
        #
        # sound = sound.numpy()
        # if len(sound.shape) > 1:
        #     if sound.shape[1] == 1:
        #         sound = sound.squeeze()
        #     else:
        #         sound = sound.mean(axis=1)  # average multiple channels
        #
        # n_fft = int(sample_rate * window_size)
        # win_length = n_fft
        # hop_length = int(sample_rate * window_stride)
        # # STFT
        # d = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length,
        #                  win_length=win_length, window=window)
        # spect, _ = librosa.magphase(d)
        # spect = np.log1p(spect)
        # spect = torch.FloatTensor(spect)
        # if normalize_audio:
        #     mean = spect.mean()
        #     std = spect.std()
        #     spect.add_(-mean)
        #     spect.div_(std)
        # return spect

    @staticmethod
    def read_audio_file(path, src_dir, sample_rate, window_size,
                        window_stride, window, normalize_audio,
                        truncate=None):
        """
        Args:
            path (str): location of a src file containing audio paths.
            src_dir (str): location of source audio files.
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
        assert (src_dir is not None) and os.path.exists(src_dir),\
            "src_dir must be a valid directory if data_type is audio"

        with codecs.open(path, "r", "utf-8") as corpus_file:
            index = 0
            for line in tqdm(corpus_file):
                audio_path = os.path.join(src_dir, line.strip())
                if not os.path.exists(audio_path):
                    audio_path = line.strip()

                assert os.path.exists(audio_path), \
                    'audio path %s not found' % (line.strip())

                if audio_path.endswith('.bin'):

                    rawdatalist, wordslist, featslist, group_data = DNADataset.extract_features(audio_path,
                                                                                                sample_rate,
                                                                                                truncate, window_size,
                                                                                                window_stride, window,
                                                                                                normalize_audio)

                    for group_raw in range(group_data):

                        rawdata = torch.from_numpy(np.array(rawdatalist[group_raw])).view(1, -1)
                        words = wordslist[group_raw]
                        feats = featslist[group_raw]

                        example_data_dict = {'src': rawdata,
                                             'src_path': line.strip(),
                                             'src_lengths': rawdata.size(1),
                                             'indices': index}

                        example_label_dict = {'tgt': words, "indices": index}

                        if feats:
                            prefix = "tgt_feat_"
                            example_label_dict.update((prefix + str(j), f)
                                                      for j, f in enumerate(feats))
                        index += 1

                        yield example_data_dict, example_label_dict

                elif audio_path.endswith('.signal'):

                    rawdatalist, group_data = DNADataset.extract_features(audio_path,
                                                                            sample_rate,
                                                                            truncate, window_size,
                                                                            window_stride, window,
                                                                            normalize_audio)

                    for group_raw in range(group_data):

                        rawdata = torch.from_numpy(np.array(rawdatalist[group_raw])).view(1, -1)

                        example_data_dict = {'src': rawdata,
                                             'src_path': line.strip(),
                                             'src_lengths': rawdata.size(1),
                                             'indices': index}

                        index += 1
                        yield example_data_dict



    @staticmethod
    def get_fields(n_src_features, n_tgt_features):
        """
        Args:
            n_src_features: the number of source features to
                create `torchtext.data.Field` for.
            n_tgt_features: the number of target features to
                create `torchtext.data.Field` for.

        Returns:
            A dictionary whose keys are strings and whose values
            are the corresponding Field objects.
        """
        fields = {}

        def make_audio(data, vocab):
            """ batch audio data """
            nfft = data[0].size(0)
            #nfft = 1
            t = max([t.size(1) for t in data])
            #t = 1
            sounds = torch.zeros(len(data), 1, nfft, t)
            for i, spect in enumerate(data):
                sounds[i, :, :, 0:spect.size(1)] = spect
            return sounds

        fields["src"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_audio, sequential=False)

        fields["src_lengths"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        for j in range(n_src_features):
            fields["src_feat_" + str(j)] = \
                torchtext.data.Field(pad_token=PAD_WORD)

        fields["tgt"] = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)

        for j in range(n_tgt_features):
            fields["tgt_feat_" + str(j)] = \
                torchtext.data.Field(init_token=BOS_WORD, eos_token=EOS_WORD,
                                     pad_token=PAD_WORD)

        fields["indices"] = torchtext.data.Field(
            use_vocab=False, dtype=torch.long,
            sequential=False)

        return fields

    @staticmethod
    def get_num_features(corpus_file, side):
        """
        For audio corpus, source side is in form of audio, thus
        no feature; while target side is in form of text, thus
        we can extract its text features.

        Args:
            corpus_file (str): file path to get the features.
            side (str): 'src' or 'tgt'.

        Returns:
            number of features on `side`.
        """
        # if side == 'src':
        #     num_feats = 0
        # else:
        #     with codecs.open(corpus_file, "r", "utf-8") as cf:
        #         f_line = cf.readline().strip().split()
        #         _, _, num_feats = DNADataset.extract_text_features(f_line)

        return 0

class DataSet(object):
    def __init__(self,
                 event,
                 event_length,
                 label,
                 label_length,
                 for_eval=False,
                 ):
        """Custruct a DataSet."""
        if for_eval == False:
            assert len(event) == len(label) and len(event_length) == len(
                label_length) and len(event) == len(
                event_length), "Sequence length for event \
            and label does not of event and label should be same, \
            event:%d , label:%d" % (len(event), len(label))
        self._event = event
        self._event_length = event_length
        self._label = label
        self._label_length = label_length
        self._reads_n = len(event)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._for_eval = for_eval
        self._perm = np.arange(self._reads_n)

    @property
    def event(self):
        return self._event

    @property
    def label(self):
        return self._label

    @property
    def event_length(self):
        return self._event_length

    @property
    def label_length(self):
        return self._label_length

    @property
    def reads_n(self):
        return self._reads_n

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def for_eval(self):
        return self._for_eval

    @property
    def perm(self):
        return self._perm

    def read_into_memory(self, index):
        event = np.asarray(list(zip([self._event[i] for i in index],
                                    [self._event_length[i] for i in index])))
        if not self.for_eval:
            label = np.asarray(list(zip([self._label[i] for i in index],
                                        [self._label_length[i] for i in index])))
        else:
            label = []
        return event, label

    def batch2sparse(label_batch):
        """Transfer a batch of label to a sparse tensor
        """
        values = []
        indices = []
        for batch_i, label_list in enumerate(label_batch[:, 0]):
            for indx, label in enumerate(label_list):
                if indx >= label_batch[batch_i, 1]:
                    break
                indices.append([batch_i, indx])
                values.append(label)
        shape = [len(label_batch), max(label_batch[:, 1])]
        return indices, values, shape

    def next_batch(self, batch_size, shuffle=True, sig_norm=False):
        """Return next batch in batch_size from the data set.
            Input Args:
                batch_size:A scalar indicate the batch size.
                shuffle: boolean, indicate if the data should be shuffled after each epoch.
            Output Args:
                inputX,sequence_length,label_batch: tuple of (indx,vals,shape)
        """
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0:
            if shuffle:
                np.random.shuffle(self._perm)
        # Go to the next epoch
        if start + batch_size > self.reads_n:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest samples in this epoch
            rest_reads_n = self.reads_n - start
            event_rest_part, label_rest_part = self.read_into_memory(
                self._perm[start:self._reads_n])

            # Shuffle the data
            if shuffle:
                np.random.shuffle(self._perm)
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_reads_n
            end = self._index_in_epoch
            event_new_part, label_new_part = self.read_into_memory(
                self._perm[start:end])
            if event_rest_part.shape[0] == 0:
                event_batch = event_new_part
                label_batch = label_new_part
            elif event_new_part.shape[0] == 0:
                event_batch = event_rest_part
                label_batch = label_rest_part
            else:
                event_batch = np.concatenate((event_rest_part, event_new_part), axis=0)
                label_batch = np.concatenate((label_rest_part, label_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            event_batch, label_batch = self.read_into_memory(
                self._perm[start:end])
        if not self._for_eval:
            label_batch = self.batch2sparse(label_batch)
        seq_length = event_batch[:, 1].astype(np.int32)
        return np.vstack(event_batch[:, 0]).astype(
            np.float32), seq_length, label_batch