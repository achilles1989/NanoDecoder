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


def ifMod(refseq, refbase, querybase):

    # motif_array = ['CCAGG','CCTGG']
    #
    # for motif in motif_array:
    #
    #     if motif in refseq or

    return True if 'CCAGG' in refseq or 'CCTGG' in refseq else False

def ifHom(refseq, refbase, querybase):

    if refbase == '.' or querybase == '.':
        tmp_array = list(refseq)
        if refbase == '.' and ([querybase,querybase] == [tmp_array[2],tmp_array[3]] or [querybase,querybase] == [tmp_array[5],tmp_array[3]] or [querybase,querybase] == [tmp_array[5],tmp_array[6]]):
            return True,False
        elif querybase == '.' and ([refbase,refbase] == [tmp_array[2],tmp_array[3]] or [refbase,refbase] == [tmp_array[5],tmp_array[3]] or [refbase,refbase] == [tmp_array[5],tmp_array[6]]):
            return False,True
    else:
        return False,False


def main():
    read_filename = sys.argv[1]

    num_mod = 0.0
    num_hom_ins = 0.0
    num_hom_del = 0.0
    num_sub = 0.0
    num_ins = 0.0
    num_del = 0.0

    length_query = 0
    length_ref = 0



    with open(read_filename, 'r') as errors:
        for line in errors:
            align_parts = line.strip().split('\t')
            if len(align_parts) < 11:
                continue

            flag = 'Sub'
            base_ref = align_parts[1]
            base_query = align_parts[2]

            mer_ref = align_parts[10]
            mer_query = align_parts[11]

            if length_query == 0:
                length_ref = int(align_parts[8])
                length_query = int(align_parts[9])

            if length_ref != int(align_parts[8]) or length_query != int(align_parts[9]):
                # print('error! changing ref or query length')
                break

            if ifMod(mer_ref, base_ref, base_query):
                num_mod += 1
                flag = 'DCM'
            elif ifHom(mer_ref, base_ref, base_query) == (True,False) :
                num_hom_ins += 1
                flag = 'HOMO_INS'
            elif ifHom(mer_ref, base_ref, base_query) == (False,True) :
                num_hom_del += 1
                flag = 'HOMO_DEL'
            elif base_query == '.':
                num_del += 1
                flag = 'DEL'
            elif base_ref == '.':
                num_ins += 1
                flag = 'INS'
            else:
                num_sub += 1

            # align_parts.append(flag)
            # print('\t'.join(align_parts))
        print('DCM\t%.5f\nHOMO_INS\t%.5f\nHOMO_DEL\t%.5f\nINS\t%.5f\nDEL\t%.5f\nSUB\t%.5f' % (num_mod/length_ref, num_hom_ins/length_ref, num_hom_del/length_ref, num_ins/length_ref, num_del/length_ref, num_sub/length_ref))

if __name__ == '__main__':
    main()
