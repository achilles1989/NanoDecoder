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

DNA_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
DNA_IDX = ['A', 'C', 'G', 'T']


def extract_fast5(input_file_path, output_path, output_prefix_feature, output_prefix_label, basecall_group, basecall_subgroup, normalization, max_length):
    """
    Extract the signal and label from a single fast5 file into Multiple Segments
    Args:
        input_file_path: path of a fast5 file.

    """

    try:
        ind_segment = 0
        (raw_data, raw_label, raw_start, raw_length) = get_label_raw(input_file_path, basecall_group, basecall_subgroup)
    except:
        return False
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
            with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature, open(
                    os.path.join(output_path, output_prefix_feature), 'a+') as file_output_feature_summary, open(
                    os.path.join(output_path, output_prefix_label), 'a+') as file_output_label:

                file_output_feature.writelines(' '.join([str(x) for x in raw_data[pre_start:raw_start[index - 1]]]))
                file_output_feature_summary.writelines('segments/'+filename_segment+'\n')
                file_output_label.writelines(' '.join([x.decode('UTF-8') for x in label_ind]) + '\n')

            ind_segment += 1
            pre_index = index - 1
            pre_start = raw_start[index - 1]

        if raw_start[index] - pre_start > max_length:
            # Skip a single event segment longer than the required signal length
            pre_index = index
            pre_start = raw_start[index]

    return True


def extract_fast5_raw(input_file_path, output_path, output_prefix, normalization, max_length, signal_stride):
    ##Open file
    try:
        ind_segment = 0
        fast5_data = h5py.File(input_file_path, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get raw data
    try:
        raw_data = list(fast5_data['/Raw/Reads/'].values())[0]
        # raw_attrs = raw_dat.attrs
        raw_data = raw_data['Signal'].value

        if normalization == 'mean':
            raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
        elif normalization == 'median':
            raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))

        for ind_segment in range(0,math.ceil(raw_data.size/signal_stride)):

            filename_segment = os.path.split(input_file_path)[1].split('.')[0] + '.' + str(ind_segment) + '.txt'

            segment_start = ind_segment*signal_stride
            segment_end = ind_segment*signal_stride + max_length
            segment_end = segment_end if segment_end < len(raw_data) else len(raw_data)

            with open(os.path.join(output_path, 'segments', filename_segment), 'w') as file_output_feature, open(
                    os.path.join(output_path, 'src', output_prefix), 'a+') as file_output_feature_summary:

                file_output_feature.writelines(' '.join([str(x) for x in raw_data[segment_start:segment_end]]))
                file_output_feature_summary.writelines('segments/' + filename_segment + '\n')

            if segment_end >= len(raw_data):
                break

        fast5_data.close()
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')
        return False

    return True


def get_label_segment(fast5_fn, basecall_group, basecall_subgroup):
    try:
        fast5_data = h5py.File(fast5_fn, 'r')
    except IOError:
        raise IOError('Error opening file. Likely a corrupted file.')

    # Get samping rate
    try:
        fast5_info = fast5_data['UniqueGlobalKey/channel_id'].attrs
        sampling_rate = fast5_info['sampling_rate'].astype('int_')
    except:
        raise RuntimeError(('Could not get channel info'))

    # Read raw data
    try:
        raw_dat = list(fast5_data['/Raw/Reads/'].values())[0]
        raw_attrs = raw_dat.attrs
    except:
        raise RuntimeError(
            'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
            'new segments cannot be identified.')
    raw_start_time = raw_attrs['start_time']

    # Read segmented data
    try:
        segment_dat = fast5_data[
            '/Analyses/' + basecall_group + '/' + basecall_subgroup + '/Events']
        segment_attrs = dict(list(segment_dat.attrs.items()))
        segment_dat = segment_dat.value

        total = len(segment_dat)

        # Process segment data
        segment_starts = segment_dat['start'] * sampling_rate - raw_start_time
        segment_lengths = segment_dat['length'] * sampling_rate
        segment_means = segment_dat['mean']
        segment_stdv = segment_dat['stdv']

        # create the label for segment event
        segment_kmer = np.full(segment_starts.shape, '-', dtype='S5')
        segment_move = np.zeros(segment_starts.shape)
        segment_cstart = np.zeros(segment_starts.shape)
        segment_clength = np.zeros(segment_starts.shape)

        segment_data = np.array(
            list(zip(segment_means, segment_stdv, segment_starts,
                     segment_lengths, segment_kmer, segment_move,
                     segment_cstart, segment_clength)),
            dtype=[('mean', 'float64'), ('stdv', 'float64'), ('start', '<u4'),
                   ('length', '<u4'), ('kmer', 'S5'),
                   ('move', '<u4'), ('cstart', '<u4'), ('clength', '<u4')])

    except:
        raise RuntimeError(
            'No events or corrupted events in file. Likely a ' +
            'segmentation error or mis-specified basecall-' +
            'subgroups (--2d?).')
    try:
        # Read corrected data
        corr_dat = fast5_data[
            '/Analyses/RawGenomeCorrected_000/' + basecall_subgroup + '/Events']
        corr_attrs = dict(list(corr_dat.attrs.items()))
        corr_dat = corr_dat.value
    except:
        raise RuntimeError((
            'Corrected data now found.'))

    corr_start_time = corr_attrs['read_start_rel_to_raw']
    corr_starts = corr_dat['start'] + corr_start_time
    corr_lengths = corr_dat['length']
    corr_bases = corr_dat['base']

    fast5_data.close()

    first_segment_index = 0
    corr_index = 2
    kmer = ''.join(corr_bases[0:5])

    # Move segment to the first available corr_data
    while segment_data[first_segment_index]['start'] < corr_starts[corr_index]:
        first_segment_index += 1

    segment_index = first_segment_index
    move = 0
    while segment_index < len(segment_data):
        my_start = corr_starts[corr_index]
        my_length = corr_lengths[corr_index]
        my_end = my_start + corr_lengths[corr_index]
        move += 1
        while True:
            segment_data[segment_index]['kmer'] = kmer
            segment_data[segment_index]['cstart'] = my_start
            segment_data[segment_index]['clength'] = my_length
            segment_data[segment_index]['move'] = move
            segment_data[segment_index]['kmer'] = kmer
            move = 0

            # if segment_data[segment_index]['start'] + segment_data[segment_index]['length'] < my_end:
            #    move = 0
            segment_index += 1
            if (segment_index >= len(segment_data)):
                break

            if segment_data[segment_index]['start'] >= my_end:
                break
            # End of while true
        corr_index += 1

        if corr_index >= len(corr_starts) - 2:
            break
        kmer = kmer[1:] + corr_bases[corr_index + 2]

    #    print first_segment_index
    #    print segment_index
    #    print corr_index
    segment_data = segment_data[first_segment_index:segment_index]
    return (segment_data, first_segment_index, segment_index, total)


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


def write_label_segment(fast5_fn, raw_label, segment_label, first, last):
    fast5_data = h5py.File(fast5_fn, 'r+')
    analyses_grp = fast5_data['/Analyses']

    label_group = "LabeledData"
    if label_group in analyses_grp:
        del analyses_grp[label_group]

    label_grp = analyses_grp.create_group(label_group)
    label_subgroup = label_grp.create_group(basecall_subgroup)

    label_subgroup.create_dataset(
        'raw_data', data=raw_data, compression="gzip")

    raw_label_data = label_subgroup.create_dataset(
        'raw_label', data=raw_label, compression="gzip")

    segment_label_data = label_subgroup.create_dataset(
        'segment_label', data=segment_label, compression="gzip")

    segment_label_data.attrs['first'] = first
    segment_label_data.attrs['last'] = last

    fast5_data.flush()
    fast5_data.close()


def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    Args:
        read (Int): An Iterable item containing element of [0,1,2,3].

    Returns:
        bpread (Char): A String containing translated dna base sequence.
    """

    base = ['A', 'C', 'G', 'T', 'M']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def add_count(concensus, start_indx, segment):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base]][start_indx + i] += 1


def simple_assembly(self,bpreads):
    concensus = np.zeros([4, 1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx, bpread in enumerate(bpreads):

        bpread = bpread.replace(' ', '')
        # bpread = bpread[0]
        if indx == 0:
            self.add_count(concensus, 0, bpread)
            continue
        d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpreads[indx]) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            census_len += 1000
        self.add_count(concensus, pos + disp, bpreads[indx])
        pos += disp
        length = max(length, pos + len(bpreads[indx]))
    return concensus[:, :length]


if __name__ == '__main__':
    fast5_fn = "/home/haotianteng/UQ/deepBNS/data/test/pass/test.fast5"

    basecall_subgroup = 'BaseCalled_template'
    basecall_group = 'RawGenomeCorrected_000'

    # Get segment data
    (segment_label, first_segment, last_segment, total) = get_label_segment(
        fast5_fn, basecall_group, basecall_subgroup)

    # segment_label is the numpy array containing labeling of the segment
    print((
        "There are {} segments, and {} are labeled ({},{})".format(total,
                                                                   last_segment - first_segment,
                                                                   first_segment,
                                                                   last_segment)))

    # get raw data
    (raw_data, raw_label, raw_start, raw_length) = get_label_raw(fast5_fn,
                                                                 basecall_group,
                                                                 basecall_subgroup)

    # You can write the labels back to the fast5 file for easy viewing with hdfviewer
    # write_label_segment(fast5_fn, raw_label, segment_label, first, last)
