# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from __future__ import absolute_import
from __future__ import print_function
import os
import h5py
import numpy as np
import math
from six.moves import zip
from statsmodels import robust
import difflib

base_keys = ['A', 'C', 'G', 'T', 'M']
base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'M': 4}


def extract_fast5(input_file_path, output_path, basecall_group, basecall_subgroup, normalization, max_length, flag_cpg, min_label):
    """
    Extract the signal and label from a single fast5 file into Multiple Segments
    Args:
        input_file_path: path of a fast5 file.

    """

    # str_summary = ''
    # str_label = ''
    dict_label = {}

    try:
        ind_segment = 0
        (raw_data, raw_label, raw_start, raw_length) = get_label_raw(input_file_path, basecall_group, basecall_subgroup)
    except:
        return dict_label
    if normalization == 'mean':
        raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
    elif normalization == 'median':
        raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))
    pre_start = raw_start[0]
    pre_index = 0
    for index, start in enumerate(raw_start):
        if start - pre_start > max_length:
            if index - 1 == pre_index:
                # If a single segment is longer than the maximum singal length, skip it.
                pre_start = start
                pre_index = index
                continue

            label_ind = raw_label['base'][pre_index:(index - 1)]

            if 'N' in label_ind:
                pre_start = start
                pre_index = index
                continue

            filename_segment = os.path.split(input_file_path)[1].split('.')[0]+'.'+str(ind_segment)+'.txt'
            if not os.path.exists(os.path.join(output_path, 'segments')):
                os.makedirs(os.path.join(output_path, 'segments'))
            # with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature, open(
            #         os.path.join(output_path, output_prefix_feature), 'a+') as file_output_feature_summary, open(
            #         os.path.join(output_path, output_prefix_label), 'a+') as file_output_label:
            with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature:

                # if 'kelvin' in filename_segment:
                #     flag_cpg = True
                # else:
                #     flag_cpg = False
                tmp_array = label_ind
                if min_label == 0 or len(label_ind) >= min_label:
                    file_output_feature.writelines(' '.join([str(x) for x in raw_data[pre_start:raw_start[index - 1]]]))
                    # str_summary += ('segments/'+filename_segment+'\n')
                    if flag_cpg:
                        for ind_x in range(len(label_ind)):
                            tmp = label_ind[ind_x].decode('UTF-8')
                            if tmp == 'C' and ind_x + 1 < len(label_ind) and label_ind[ind_x + 1].decode('UTF-8') == 'G':
                                tmp_array[ind_x] = 'M'

                    # str_label += (' '.join([x.decode('UTF-8') for x in tmp_array]) + '\n')
                    dict_label['segments/'+filename_segment]=' '.join([x.decode('UTF-8') for x in tmp_array])
                    # file_output_feature_summary.writelines('segments/'+filename_segment+'\n')
                    # file_output_label.writelines(' '.join([x.decode('UTF-8') for x in label_ind]) + '\n')

            ind_segment += 1
            pre_index = index - 1
            pre_start = raw_start[index - 1]

        if raw_start[index] - pre_start > max_length:
            # Skip a single event segment longer than the required signal length
            pre_index = index
            pre_start = raw_start[index]

    # return [str_summary, str_label]
    return dict_label


def extract_signal(input_file_path, output_path, basecall_group, basecall_subgroup, normalization, max_length, flag_cpg, min_label):
    """
    Extract the signal and label from a single fast5 file into Multiple Segments
    Args:
        input_file_path: path of a fast5 file.

    """

    str_summary = ''
    str_label = ''
    dict_label = {}

    try:
        ind_segment = 0
        file_raw = open(input_file_path,'r')
        raw_data = [int(x) for x in file_raw.read().strip().split()]
        file_label = open(os.path.splitext(input_file_path)[0]+'.label','r')
        raw_label_array = file_label.read().strip().split('\n')
        raw_start = []
        raw_length = []
        raw_label = []
        for tmp_label in raw_label_array:
            tmp_array = tmp_label.split()
            raw_start.append(int(tmp_array[0]))
            raw_length.append(int(tmp_array[1]) - int(tmp_array[0]) + 1)
            raw_label.append(tmp_array[2])
        label_data = np.array(
            list(zip(raw_start, raw_length, raw_label)),
            dtype=[('start', '<u4'), ('length', '<u4'), ('base', 'S1')])
    except:
        return dict_label
    if normalization == 'mean':
        raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
    elif normalization == 'median':
        raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))
    pre_start = raw_start[0]
    pre_index = 0
    for index, start in enumerate(raw_start):
        if start - pre_start > max_length:

            if index - 1 == pre_index:
                # If a single segment is longer than the maximum singal length, skip it.
                pre_start = start
                pre_index = index
                continue

            label_ind = label_data['base'][pre_index:(index - 1)]

            if 'N' in label_ind:
                pre_start = start
                pre_index = index
                continue

            filename_segment = os.path.split(input_file_path)[1].split('.')[0]+'.'+str(ind_segment)+'.txt'
            if not os.path.exists(os.path.join(output_path, 'segments')):
                os.makedirs(os.path.join(output_path, 'segments'))
            # with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature, open(
            #         os.path.join(output_path, output_prefix_feature), 'a+') as file_output_feature_summary, open(
            #         os.path.join(output_path, output_prefix_label), 'a+') as file_output_label:
            with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature:

                tmp_array = label_ind

                if min_label == 0 or len(label_ind) >= min_label:
                    file_output_feature.writelines(' '.join([str(x) for x in raw_data[pre_start:raw_start[index - 1]]]))

                    if flag_cpg:
                        for ind_x in range(len(label_ind)):
                            tmp = label_ind[ind_x].decode('UTF-8')
                            if tmp == 'C' and ind_x + 1 < len(label_ind) and label_ind[ind_x + 1].decode('UTF-8') == 'G':
                                tmp_array[ind_x] = 'M'

                    # str_summary += ('segments/'+filename_segment+'\n')
                    # str_label += (' '.join([x.decode('UTF-8') for x in label_ind]) + '\n')
                    # file_output_feature_summary.writelines('segments/'+filename_segment+'\n')
                    # file_output_label.writelines(' '.join([x.decode('UTF-8') for x in label_ind]) + '\n')
                    dict_label['segments/' + filename_segment] = ' '.join([x.decode('UTF-8') for x in tmp_array])

            ind_segment += 1
            pre_index = index - 1
            pre_start = raw_start[index - 1]

        if raw_start[index] - pre_start > max_length:
            # Skip a single event segment longer than the required signal length
            pre_index = index
            pre_start = raw_start[index]

    return dict_label


def extract_fast5_raw(input_file_path, output_prefix, normalization, max_length, signal_stride, suffix):

    result_array = [output_prefix]

    ##Open file
    if suffix == 'fast5' :
        try:
            fast5_data = h5py.File(input_file_path, 'r')
        except IOError:
            raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:

        if suffix == 'fast5':

            raw_data = list(fast5_data['/Raw/Reads/'].values())[0]
            # raw_attrs = raw_dat.attrs
            raw_data = raw_data['Signal'].value

        else:

            tmp = open(input_file_path,'r').read()
            raw_data = [float(x) for x in tmp.strip().split()]

        raw_data = np.array(raw_data)
        if normalization == 'mean':
            raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
        elif normalization == 'median':
            raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))

        for ind_segment in range(0, math.ceil(raw_data.size / signal_stride)):

            segment_start = ind_segment * signal_stride
            segment_end = ind_segment * signal_stride + max_length
            segment_end = segment_end if segment_end < len(raw_data) else len(raw_data)

            result_array.append(' '.join([str(x) for x in raw_data[segment_start:segment_end]]))
            if segment_end >= len(raw_data):
                break

        if suffix == 'fast5':
            fast5_data.close()
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')
        return result_array

    return result_array


def get_label_raw(fast5_fn, basecall_group, basecall_subgroup):
    ##Open file
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        # raw_attrs = raw_dat.attrs
        raw_dat = raw_dat['Signal'].value
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')

    # Read corrected data
    try:
        corr_data = fast5_data[
            '/Analyses/'+basecall_group +'/' + basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_data.attrs.items()))
        corr_data = corr_data.value
    except:
        raise RuntimeError((
            'Corrected data not found.'))

    # fast5_info = fast5_data['UniqueGlobalKey/channel_id'].attrs
    # sampling_rate = fast5_info['sampling_rate'].astype('int_')

    # Reading extra information
    corr_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']  #
    if len(raw_dat) > 99999999:
        raise ValueError(fast5_fn + ": max signal length exceed 99999999")
    if any(len(vals) <= 1 for vals in (
            corr_data, raw_dat)):
        raise NotImplementedError((
            'One or no segments or signal present in read.'))
    event_starts = corr_data['start'] + corr_start_rel_to_raw
    event_lengths = corr_data['length']
    event_bases = corr_data['base']

    fast5_data.close()
    label_data = np.array(
        list(zip(event_starts, event_lengths, event_bases)),
        dtype=[('start', '<u4'), ('length', '<u4'), ('base', 'S1')])
    return (raw_dat, label_data, event_starts, event_lengths)


def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    Args:
        read (Int): An Iterable item containing element of [0,1,2,3].

    Returns:
        bpread (Char): A String containing translated dna base sequence.
    """

    bpread = [base_keys[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def add_count(concensus, start_indx, segment):
    # base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'M': 4, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'm': 4}

    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base.upper()]][start_indx + i] += 1


def simple_assembly(bpreads, flag_intersection=True):

    if flag_intersection:
        concensus = np.zeros([len(base_keys), 1000])
        pos = 0
        length = 0
        census_len = 1000
        bpreads_valid = [x[0].replace(' ', '') for x in bpreads if x[0] != '']
        bpreads = bpreads_valid

        for indx, bpread in enumerate(bpreads):

            # bpread = bpread[0]
            if indx == 0:
                add_count(concensus, 0, bpread)
                continue

            d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
            match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
            disp = match_block[0] - match_block[1]
            if disp + pos + len(bpreads[indx]) > census_len:
                concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                       mode='constant', constant_values=0)
                census_len += 1000
            add_count(concensus, pos + disp, bpreads[indx])
            pos += disp
            length = max(length, pos + len(bpreads[indx]))
        return concensus[:, :length]

    else:

        bpreads_valid = [x[0].replace(' ', '') for x in bpreads if x[0] != '']
        return ''.join(bpreads_valid)
