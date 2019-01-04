import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import axes
import configargparse
import os
import math


def config_opts():
    parser = configargparse.ArgumentParser(
        description='plotAttenHeat.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add('-path', '--path', required=True,
               help='attention root path')
    #parser.add('-start', '--start', required=False,
    #           help='start point')
    #parser.add('-end', '--end', required=False,
    #           help='end point')
    parser.add('-max', '--max', required=True,
               help='attention threshold')
    parser.add('-save', '--save', required=True,
               help='save root path')
    return parser

if __name__ == '__main__':

    parser = config_opts()
    opt = parser.parse_args()

    #tmp = np.fromfile(opt.path)
    #print(tmp.size())

    list_label = list()
    list_signal = list()
    list_atten = list()
    dict_atten = dict()
    list_segment = list()

    for file in os.listdir(opt.path):
        with open(os.path.join(opt.path,file)) as file_atten:
            while True:
                line = file_atten.readline().strip()
                if line:
                    tmp_array = line.split()
                    if tmp_array[0] == '>':
                        # print(len(list_label))
                        # print(len(dict_atten[len(dict_atten)]))
                        if len(list_label) != 0 and len(list_label)-list_segment[len(dict_atten)-1][0] != len(dict_atten[len(dict_atten)-1]):
                            list_label.pop(len(list_label)-1)
                            #if len(list_label) != 0:
                            #print(len(list_label)-list_segment[len(dict_atten)-1][0])
                            #print(len(dict_atten[len(dict_atten)-1]))
                        list_segment.append([len(list_label),len(list_signal)])
                        list_signal.extend(tmp_array[1:tmp_array.index('|')])
                        list_label.extend(tmp_array[tmp_array.index('|')+1:])
                        dict_atten[len(dict_atten)] = list()
                    else:

                        dict_atten[len(dict_atten)-1].append([float(x) for x in tmp_array])
                        #list_atten.append(tmp_array)
                else:
                    break

    #len_signal = len(list_signal)
    #len_label = len(list_label)

    #list_atten = np.zeros((len_label,len_signal))
    #print(list_atten.shape)

    # fig = plt.figure()

    def draw_heatmap(data, xlabels, ylabels, save):
        # cmap=cm.Blues
        cmap = cm.get_cmap('rainbow', 1000)
        figure = plt.figure(facecolor='w')
        ax = figure.add_subplot(1, 1, 1, position=[0.1, 0.15, 0.8, 0.8])
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        vmax = data[0][0]
        vmin = data[0][0]
        for i in data:
            for j in i:
                if j > vmax:
                    vmax = j
                if j < vmin:
                    vmin = j
        map = ax.imshow(data, interpolation='nearest', cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        cb = plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
        plt.savefig(save)
        #plt.show()

    #for index_segment in range(1,len(list_segment)):
    #    pre_segment = list_segment[index_segment-1]
    #    segment = list_segment[index_segment]

    #    list_atten[pre_segment[0]:segment[0], pre_segment[1]:segment[1]] = np.array(dict_atten[index_segment - 1])
        #data_plot = np.array(dict_atten[index_segment - 1])
        #print(data_plot)
        #draw_heatmap(data_plot,list_signal[pre_segment[1]:segment[1]],list_label[ pre_segment[0]:segment[0]], opt.save+'/atten'+str(index_segment)+'.jpg')
        #label_signal = list()
        #for tmp in range(segment[1]-pre_segment[1]):
        #    label_signal.append('')
        #label_signal[0] = str(pre_segment[1])
        #label_signal[-1] = str(segment[1])
        #draw_heatmap(data_plot, label_signal, list_label[pre_segment[0]:segment[0]],
        #opt.save + '/atten' + str(index_segment) + '.jpg')

    #data_plot = 100 * list_atten[int(opt.start):int(opt.end),1:15000]
    #print(data_plot)
    #draw_heatmap(data_plot,list_signal[1:15000],list_label[int(opt.start):int(opt.end)],opt.save+'/atten'+str(index_segment)+'.jpg')
    # plt.figure()
    # fig,ax = plt.subplots()
    # im = ax.imshow(data_plot)
    # plt.imshow(data_plot)
    # plt.grid(True)
    # plt.savefig("/media/quanc/E/data/chiron_train/res_pytorch/data_10000_12000/raw_opennmt/test/1.jpg")
    # plt.show()


    dict_kmer = dict()

    for index_segment in range(1,len(list_segment)):

        pre_segment = list_segment[index_segment-1]
        segment = list_segment[index_segment]

        tmp_label = list_label[pre_segment[0]:segment[0]]
        tmp_atten = np.array(dict_atten[index_segment - 1])
        tmp_signal = list_signal[pre_segment[1]:segment[1]]


        if len(tmp_label) < 5:
            continue

        # print(tmp_label)
        for index_label in range(5,len(tmp_label)+1):
            #print(''.join(list_label[index_label-5:index_label]))
            str_kmer = ''.join(tmp_label[index_label-5:index_label])

            if str_kmer.__contains__('</s>'):
                continue

            # print(str_kmer)
            if str_kmer not in dict_kmer:
                dict_kmer[str_kmer] = list()
                # print(str_kmer)

            # print(tmp_signal)
            # print(np.argsort(tmp_atten[index_label-5:index_label,],axis=1))
            tmp_index = np.argsort(tmp_atten[index_label-5:index_label,],axis=1)[:,-int(opt.max):]

            list_index = list()

            for array in tmp_index:
                list_index.extend(array)

            list_index = list(set(list_index))
            list_index.sort()
            #print(list_index)

            # tmp_index = list(set([x for x in np.where(tmp_atten[index_label-5:index_label,] > 0)[1]]))
            # print(tmp_index)
            # print(len(tmp_index))
            # print(max(tmp_atten[np.where(tmp_atten[index_label-5:index_label,] > 0)]))
            # print(min(tmp_atten[np.where(tmp_atten[index_label - 5:index_label, ] > 0)]))
            # print([float(tmp_signal[x]) for x in tmp_index])
            dict_kmer[str_kmer].extend([float(tmp_signal[x]) for x in list_index])

    with open(os.path.join(opt.save,'5mer.txt'),'w') as file_kmer:
        for key,val in dict_kmer.items():
            #print(key)
            #print(val)
            file_kmer.write('%s\t%0.5f\t%0.5f\t%d\n' % (
                key,
                np.median(val),
                np.std(val),
                len(val)
            ))















