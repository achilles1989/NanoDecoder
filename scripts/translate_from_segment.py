import os
import configargparse
from utils.labelop import index2base, add_count
import numpy as np
import difflib

def config_opts():
    parser = configargparse.ArgumentParser(
        description='translate_from_segment.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('-src', '--source', required=True,
               help='traslated segments')
    parser.add('-o', '--output', required=True,
               help='save path')
    return parser

def simple_assembly(bpreads, flag_intersection=True):

    if flag_intersection:
        concensus = np.zeros([4, 1000])
        pos = 0
        length = 0
        census_len = 1000
        bpreads_valid = [x.replace(' ', '') for x in bpreads if x != '']
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

        bpreads_valid = [x.replace(' ', '') for x in bpreads if x != '']
        return ''.join(bpreads_valid)

if __name__ == '__main__':
    parser = config_opts()
    opt = parser.parse_args()

    with open(opt.output,'w+') as file_out:
        for path_segment in os.listdir(opt.source):
            with open(os.path.join(opt.source,path_segment),'r') as file_segment:
                line = file_segment.read().strip().split('\n')
                c_bpread = index2base(np.argmax(simple_assembly(line), axis=0))
                # c_bpread = simple_assembly(line, flag_intersection=False)

                file_out.writelines('>%s\t%d bps\t%d segments\n%s\n' % (path_segment.split('.')[0],len(c_bpread),len(line),c_bpread))


