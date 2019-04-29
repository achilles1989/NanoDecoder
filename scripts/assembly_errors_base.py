#!/usr/bin/env python3
"""
Inputs:
  *

Output:
  * tsv file with these columns: length, identity, relative length

"""

import collections
import gzip
import os
import statistics
import sys

def main():
    read_filename = sys.argv[1]
    paf_filename = sys.argv[2]

    dict_length = {}

    with open(paf_filename, 'r') as source:
        for line in source:
            align_parts = line.strip().split('\t')
            if align_parts[1] not in dict_length:
                dict_length[align_parts[1]]=int(align_parts[10])

    num_ins_a = 0.0
    num_del_a = 0.0
    num_sub_a = 0.0
    num_ins_g = 0.0
    num_del_g = 0.0
    num_sub_g = 0.0
    num_ins_c = 0.0
    num_del_c = 0.0
    num_sub_c = 0.0
    num_ins_t = 0.0
    num_del_t = 0.0
    num_sub_t = 0.0
    num_ins_m = 0.0
    num_del_m = 0.0
    num_sub_m = 0.0
    num_error = 0.0

    query_id = ''
    length_ref_save = dict()

    with open(read_filename, 'r') as errors:
        for line in errors:
            align_parts = line.strip().split('\t')
            if len(align_parts) < 11:
                continue

            if align_parts[9] not in dict_length:
                continue
            base_ref = align_parts[1]
            base_query = align_parts[2]

            if align_parts[15] not in length_ref_save:

                if len(length_ref_save) > 0:
                    # print('\t'.join(['{:.5f}'.format(x/length_ref_save[query_id][0]) for x in [num_ins_a,num_del_a,num_sub_a,num_ins_g,num_del_g,num_sub_g,num_ins_c,num_del_c,num_sub_c,num_ins_t,num_del_t,num_sub_t,num_ins_m,num_del_m,num_sub_m]])+'\t'+'\t'.join(str(x) for x in length_ref_save[query_id])+'\t%.5f'%(num_error/length_ref_save[query_id][0]))

                    print((15 * '%.5f\t%s\t%s\t%d\t%d\n')[:-1]
                          % (num_ins_a/length_ref_save[query_id][0],'A','INS',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_del_a/length_ref_save[query_id][0],'A','DEL',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_sub_a/length_ref_save[query_id][0],'A','SUB',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_ins_g/length_ref_save[query_id][0],'G','INS',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_del_g/length_ref_save[query_id][0],'G','DEL',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_sub_g/length_ref_save[query_id][0],'G','SUB',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_ins_c/length_ref_save[query_id][0],'C','INS',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_del_c/length_ref_save[query_id][0],'C','DEL',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_sub_c/length_ref_save[query_id][0],'C','SUB',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_ins_t/length_ref_save[query_id][0],'T','INS',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_del_t/length_ref_save[query_id][0],'T','DEL',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_sub_t/length_ref_save[query_id][0],'T','SUB',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_ins_m/length_ref_save[query_id][0],'M','INS',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_del_m/length_ref_save[query_id][0],'M','DEL',length_ref_save[query_id][0],length_ref_save[query_id][1],
                             num_sub_m/length_ref_save[query_id][0],'M','SUB',length_ref_save[query_id][0],length_ref_save[query_id][1],))

                num_ins_a = 0.0
                num_del_a = 0.0
                num_sub_a = 0.0
                num_ins_g = 0.0
                num_del_g = 0.0
                num_sub_g = 0.0
                num_ins_c = 0.0
                num_del_c = 0.0
                num_sub_c = 0.0
                num_ins_t = 0.0
                num_del_t = 0.0
                num_sub_t = 0.0
                num_ins_m = 0.0
                num_del_m = 0.0
                num_sub_m = 0.0
                num_error = 0.0

                # length_ref_save[align_parts[15]]=[int(align_parts[9]),int(align_parts[8])]
                length_ref_save[align_parts[15]]=[dict_length[align_parts[9]],int(align_parts[8])]
                query_id = align_parts[15]

            num_error += 1

            if base_ref == 'A':
                if base_query == '.':
                    num_del_a += 1
                else:
                    num_sub_a += 1
            elif base_ref == 'G':
                if base_query == '.':
                    num_del_g += 1
                else:
                    num_sub_g += 1
            elif base_ref == 'C':
                if base_query == '.':
                    num_del_c += 1
                else:
                    num_sub_c += 1
            elif base_ref == 'T':
                if base_query == '.':
                    num_del_t += 1
                else:
                    num_sub_t += 1
            elif base_ref == 'M':
                if base_query == '.':
                    num_del_m += 1
                else:
                    num_sub_m += 1
            elif base_ref == '.':
                if base_query == 'A':
                    num_ins_a += 1
                elif base_query == 'G':
                    num_ins_g += 1
                elif base_query == 'C':
                    num_ins_c += 1
                elif base_query == 'T':
                    num_ins_t += 1
                elif base_query == 'M':
                    num_ins_m += 1

        print((15 * '%.5f\t%s\t%s\t%d\t%d\n')[:-1]
              % (num_ins_a / length_ref_save[query_id][0], 'A', 'INS', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_del_a / length_ref_save[query_id][0], 'A', 'DEL', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_sub_a / length_ref_save[query_id][0], 'A', 'SUB', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_ins_g / length_ref_save[query_id][0], 'G', 'INS', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_del_g / length_ref_save[query_id][0], 'G', 'DEL', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_sub_g / length_ref_save[query_id][0], 'G', 'SUB', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_ins_c / length_ref_save[query_id][0], 'C', 'INS', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_del_c / length_ref_save[query_id][0], 'C', 'DEL', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_sub_c / length_ref_save[query_id][0], 'C', 'SUB', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_ins_t / length_ref_save[query_id][0], 'T', 'INS', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_del_t / length_ref_save[query_id][0], 'T', 'DEL', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_sub_t / length_ref_save[query_id][0], 'T', 'SUB', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_ins_m / length_ref_save[query_id][0], 'M', 'INS', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_del_m / length_ref_save[query_id][0], 'M', 'DEL', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],
                 num_sub_m / length_ref_save[query_id][0], 'M', 'SUB', length_ref_save[query_id][0],
                 length_ref_save[query_id][1],))


if __name__ == '__main__':
    main()
